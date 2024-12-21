import copy
import math
import pprint

class Form(object):
    """
    The base class for all tax forms. This acts like a dictionary but has
    some useful behaviors:
      - Values assigned in the dictionary are automatically rounded to the
        nearest integer (the "whole dollar method").
      - When missing values are retrieved with the [] operator, 0 is returned.
      - Individual lines in the form can be automatically initialized by the
        "inputs" dictionary.
    """
    
    def __init__(self, init_idx=None, **kwargs):
        self.data = {}
        self.comment = {}
        self.must_file = False
        self.forms = []
        self.disable_rounding = kwargs.get('disable_rounding', False)
        name = self.__class__.__name__
        if name in kwargs:
            if init_idx is not None:
                init_dict = kwargs[name][init_idx]
            else:
                init_dict = kwargs[name]
            for key in init_dict:
                self[key] = init_dict[key]

    def addForm(self, form):
        """
        Add another form as a child of this form. This is for later use
        by the printAllForms() method.
        """
        self.forms.append(form)

    def get(self, key):
        return self.data[key] if key in self.data else None
    def __getitem__(self, key):
        val = self.data.get(key)
        return 0 if val is None else val

    def __contains__(self, key):
        return key in self.data

    def __setitem__(self, key, val):
        if val is None:
            if key in self.data:
                del self.data[key]
        elif self.disable_rounding:
            self.data[key] = float(val)
        else:
            self.data[key] = int(round(val))

    def rowSum(self, rows):
        """
        Convenience function for summing a list of rows in the form by
        regex pattern. If all named rows are blank, returns None.
        """
        raw_pattern = '(' + '|'.join(rows) + ')'
        pattern = re.compile(raw_pattern)
        val = sum(v for k, v in self.data.items() if pattern.fullmatch(k))
        if any(pattern.fullmatch(k) for k, v in self.data.items()):
            return val
        return None

    def spouseSum(self, inputs, field):
        """
        Sums two spouses inputs if filing a joint return.
        """
        if field not in inputs:
            return None
        if inputs['status'] == "joint":
            return inputs[field][0] + inputs[field][1]
        else:
            return inputs[field]

    def printForm(self):
        """
        Prints all rows of a form, skipping empty rows. The rows are sorted
        sensibly.
        """
        def atoi(s):
            return int(s) if s.isdigit() else s
        def mixed_keys(s):
            return [ atoi(c) for c in re.split('(\d+)', s) ]

        locale.setlocale(locale.LC_ALL, '')
        print('%s:' % self.title())
        keys = list(self.data.keys())
        keys = sorted(keys, key=mixed_keys)
        for k in keys:
            print('  %6s %11s' % (k, locale.format('%d', self[k], 1)), end='')
            if k in self.comment:
                print('  %s' % self.comment[k], end='')
            print('')

    def printAllForms(self):
        """
        Prints all child forms of this form that must be filed with the
        tax return.
        """
        for f in self.forms:
            if f.mustFile():
                f.printForm()




class F8889(Form):
    pass



class F1040SA(Form):
    """
    Schedule A (Form 1040), Itemized Deductions
    """

    def __init__(f, inputs, f1040):
        super(F1040SA, f).__init__(inputs)
        if f['1']:
            f['2'] = f1040['11']
            f['3'] = f['2'] * .075
            f['4'] = max(0, f['1'] - f['3'])
        f['5a'] = inputs['state_withholding'] + \
                  inputs.get('extra_state_tax_payments', 0)
        f['5d'] = f.rowSum(['5a', '5b', '5c'])
        f['5e'] = min(f['5d'], 5000 if inputs['status'] == 'separate' else 10000)
        f['7'] = f.rowSum(['5e', '6'])
        f['8e'] = f.rowSum(['8a', '8b', '8c', '8d'])
        f['10'] = f.rowSum(['8e', '9'])
        f['14'] = f.rowSum(['11', '12', '13'])
        f['17'] = f.rowSum(['4', '7', '10', '14', '15', '16'])

    def title(self):
        return 'Schedule A (Form 1040)'



class F1040SD(Form):
    """
    Schedule D (Form 1040), Capital Gains and Losses
    """

    def __init__(f, **inputs):
        super(F1040SD, f).__init__(inputs)
        if 'capital_gain_long' not in inputs \
            and 'capital_gain_short' not in inputs \
            and 'capital_gain_carryover_short' not in inputs \
            and 'capital_gain_carryover_long' not in inputs:
            return

        f.must_file = True

        # Part I: Short-Term Capital Gains and Losses—Generally Assets Held One Year or Less 
        f.comment['1a'] = 'Short-term capital gain'
        f['1a'] = inputs.get('capital_gain_short')

        f.comment['6'] = 'Short-term capital gain carryover'
        f['6'] = inputs.get('capital_gain_carryover_short')
        
        f.comment['7'] = 'Net short-term capital gain or (loss)'
        f['7'] = f.rowSum(['1a', '1b', '2', '3', '4', '5', '6'])

        # Part II: Long-Term Capital Gains and Losses—Generally Assets Held More Than One Year
        f.comment['8a'] = 'Long-term capital gain'
        f['8a'] = inputs.get('capital_gain_long')
        
        f.comment['13'] = 'Capital gain distributions'
        f['13'] = inputs.get('capital_gain_distribution')

        f.comment['14'] = 'Long-term capital gain carryover'
        f['14'] = inputs.get('capital_gain_carryover_long')
        
        f.comment['15'] = 'Net long-term capital gain or (loss)'
        f['15'] = f.rowSum(['8a', '8b', '9', '10', '11', '12', '13', '14'])
        
        # Part III: Summary
        f['16'] = f.rowSum(['7', '15'])

        if f['16'] < 0:
            cutoff = -1500 if inputs['status'] == "separate" else -3000
            f['21'] = max(f['16'], cutoff)

        # if lines 15 and 16 are both gains and line 18 or 19 has a value:
        #     Use the Schedule D tax worksheet
        # else if lines 15 and 16 are both gains or you have qualified divs:
        #     Use the Qualified Dividends and Capital Gain Tax Worksheet
        # else
        #     Use tax tables

    @property
    def title(f):
        return 'Schedule D (Form 1040)'



class F1040S1(Form):
    """
    Schedule 1 (Form 1040): Additional Income and Adjustments to Income
    """

    def __init__(f, **inputs):
        super(F1040S1, f).__init__(inputs)

        f.must_file = True

        # Part I: Additional Income
        f.comment['1'] = "Taxable refunds, credits, or offsets of state and local income taxes"
        f['1'] = inputs.get('state_refund_taxable')

        f.comment['3'] = 'Business income or (loss)'
        f['3'] = f.spouseSum(inputs, 'business_income')

        f.comment['7'] = 'unemployment_compensation'
        f['7'] = inputs.get('unemployment_compensation')

        f.comment['9'] = 'Total other income'
        f['9'] = f.rowSum(['8[a-z]'])

        f.comment['10'] = 'Additional income'
        f['10'] = f.rowSum(['1', '2a', '[3-7]', '9'])

        # Part II: Adjustments to Income
        f.comment['25'] = 'Total other adjustments.'
        f['25'] = f.rowSum(['24[a-z]'])

        f.comment['26'] = 'adjustments to income'
        f['26'] = f.rowSum(['1[1-8]', '19a', '2[0-3]', '25'])

    @property
    def title(self):
        return 'Schedule 1 (Form 1040)'



class F1040(Form):
    """
    Form 1040
    """

    def __init__(f, **inputs):
        super(F1040, f).__init__(inputs)

        with open(f"{inputs['status']}_{inputs['year']}_tax_info.json", "r") as file:
            f.info = json.load(file)

        f.must_file = True
        f.addForm(f)

        f.comment['1a'] = 'Wages: Total amount from Form(s) W-2, box 1'
        f['1a'] = f.spouseSum(inputs, 'wages')

        f['1z'] = f.rowSum(['1a', '1b', '1c', '1d', '1e', '1f', '1g', '1h'])

        f.comment['2a'] = 'Tax-exempt interest'
        f['2a'] = inputs.get('tax_exempt_interest')

        f.comment['2b'] = 'Taxable Interest'
        f['2b'] = inputs.get('taxable_interest')

        f.comment['3a'] = 'Qualified dividends'
        f['3a'] = inputs.get('qualified_dividends')

        f.comment['3b'] = 'Ordinary dividends'
        f['3b'] = inputs.get('dividends')

        sd = F1040SD(inputs)
        f.addForm(sd)

        f.comment['7'] = 'Capital gain or (loss)'
        f['7'] = sd['21'] or sd['16'] if sd.must_file else inputs.get('capital_gain_distribution')

        s1 = F1040S1(inputs)
        f.addForm(s1)

        f.comment['8'] = 'Additional income from Schedule 1, line 10'
        f['8'] = s1['10']

        f.comment['9'] = 'Total Income'
        f['9'] = f.rowSum(['1z', '2b', '3b', '4b', '5b', '6b', '7', '8'])

        f.comment['10'] = 'Adjustments to income from Schedule 1, line 26'
        f['10'] = s1['26']

        f.comment['11'] = 'Adjusted gross income'
        f['11'] = f['9'] - f['10']

        sa = F1040SA(inputs, f)
        f.addForm(sa)
        sa.must_file = inputs.get('itemize_deductions', sa['17'] > f.info['standard deduction'])

        f.comment['12'] = 'Itemized deductions' if sa.must_file else 'Standard deduction'
        f['12'] = sa['17'] if sa.must_file else f.info['standard deduction']

        f['14'] = f.rowSum(['12', '13'])

        f.comment['15'] = 'Taxable Income'
        f['15'] = max(0, f['11'] - f['14'])

        f.comment['16'] = 'Tax'
        f['16'] = f.qualfiedDividendsAndCapitalGainTaxWorksheet(inputs, sd)['25']

        f['18'] = f.rowSum(['16', '17'])

        f.comment['21'] = 'Total Credits'
        f['21'] = f['19'] + f['20']

        f['22'] = max(0, f['18'] - f['21'])

        f.comment['24'] = 'Total Tax'
        f['24'] = f.rowSum(['22', '23'])

        f['25a'] = inputs.get('withheld_fed', 0)

        f['25d'] = f.rowSum(['25a', '25b', '25c'])

        f.comment['33'] = 'Total payments'
        f['33'] = f.rowSum(['25d', '26', '32'])

        if f['33'] > f['24']:
            f.comment['34'] = 'Refund'
            f['34'] = f['33'] - f['24']
        else:
            f.comment['37'] = 'Amount you owe'
            f['37'] = f['24'] - f['33']


    def qualfiedDividendsAndCapitalGainTaxWorksheet(f, inputs, sd):
        """
        Qualified Dividends and Capital Gain Tax Worksheet
        """
        w = {}
        w['1'] = f['15'] # Taxable income
        w['2'] = f['3a'] # Qualified dividends
        w['3'] = max(0, min(sd['15'], sd['16'])) if sd.must_file else f['7'] # Capital gain or (loss)
        w['4'] = w['2'] + w['3']
        w['5'] = max(0, w['1'] - w['4'])
        w['6'] = f.CAPGAIN15_LIMITS
        w['7'] = min(w['1'], w['6'])
        w['8'] = min(w['5'], w['7'])
        w['9'] = w['7'] - w['8']
        w['10'] = min(w['1'], w['4'])
        w['11'] = w['9']
        w['12'] = w['10'] - w['11']
        w['13'] = f.CAPGAIN20_LIMITS
        w['14'] = min(w['1'], w['13'])
        w['15'] = w['5'] + w['9']
        w['16'] = max(0, w['14'] - w['15'])
        w['17'] = min(w['12'], w['16'])
        w['18'] = w['17'] * 0.15
        w['19'] = w['9'] + w['17']
        w['20'] = w['10'] - w['19']
        w['21'] = w['20'] * 0.20
        w['22'] = f.taxComputationWorksheet(*f.info["federal income tax bracket"], w['5'])
        w['23'] = w['18'] + w['21'] + w['22']
        w['24'] = f.taxComputationWorksheet(*f.info["federal income tax bracket"], w['1'])
        w['25'] = min(w['23'], w['24'])
        return w

    @staticmethod
    def taxComputationWorksheet(interval, rates, income, staircase=True):
        """
        Tax Computation Worksheet
        """
        assert len(interval) == len(rates)

        if not staircase:
            for i in range(len(rates)-1):
                if interval[i] < income <= interval[i+1]:
                    return income * rates[i]
            return income * rates[-1]

        tax = 0
        for i in range(len(rates)-1):
            if income < interval[i+1]:
                return tax + (income - interval[i]) * rates[i]
            else:
                tax += (interval[i+1] - interval[i]) * rates[i]
        return tax + (income - interval[-1]) * rates[-1]

    @property
    def title(f):
        return 'Form 1040'