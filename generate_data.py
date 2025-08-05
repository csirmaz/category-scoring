
import numpy as np
import math


class GenerateData:
    
    CAT_HOME = 0
    CAT_HOSPITAL = 1
    CAT_DEATH = 2
    
    def rnd_inputs(self):
        """Return a random set of inputs"""
        r = np.random.random_sample(size=9)
        age = r[5] * 100.
        return {
            'prev_infection_a': 1 if r[0] > .5 else 0,
            'prev_infection_b': 1 if r[1] > .5 else 0,
            'acute_infection_b': 1 if r[2] > .5 else 0,
            'cancer_diagnosis': 1 if r[3] > .5 else 0,
            'weight_deviation': r[4] * 200. - 100.,
            'age': age,
            'blood_pressure_dev': r[6] * 100.,
            'smoked_years': 0 if r[7] < .5 else (r[8] * (age - 16.) + 2.)
        }
    
    def normalize_inputs(self, inp):
        """Make the inputs zero mean unit variance. See test_inputs_mean_variance() for measuring mean, var for smoked_years"""
        return [
            (inp['prev_infection_a'] - .5) / math.sqrt(.25),
            (inp['prev_infection_b'] - .5) / math.sqrt(.25),
            (inp['acute_infection_b'] - .5) / math.sqrt(.25),
            (inp['cancer_diagnosis'] - .5) / math.sqrt(.25),
            (inp['weight_deviation'] - 0) / math.sqrt(3332.),
            (inp['age'] - 50.) / math.sqrt(833.),
            (inp['blood_pressure_dev'] - 50.) / math.sqrt(833.),
            (inp['smoked_years'] - 9.5) / math.sqrt(278.)
        ]
    
    
    def input_to_category(self, inp):
        """Implements the toy dataset. In a deterministic fashion, return a category
        given a set of inputs.
        
            inp = {
                'prev_infection_a':  bool   # Previous infection with pathogen A
                'prev_infection_b':  bool   # Previous infection with pathogen B
                'acute_infection_b': bool   # Acute / current infection with pathogen B
                'cancer_diagnosis':  bool   # Preexisting cancer diagnosis
                'weight_deviation':  -100..100  # Deviation of body weight
                'age':                  0..100  # Age (years)
                'blood_pressure_dev':   0..100  # Blood pressure deviation, 100=severe hypertension
                'smoked_years':         0..     # Years smoked
            }
            
            Outcomes / categories:
                - CAT_HOME     No hospital treament
                - CAT_HOSPITAL Hospital admission
                - CAT_DEATH    Death
        """
        
        virtual_age = inp['age'] + inp['smoked_years'] * 1.2 + inp['weight_deviation'] * 0.1
        immune_overdrive = inp['cancer_diagnosis'] * 10 - inp['smoked_years'] * 2 - inp['age'] * .1 + inp['prev_infection_b'] * 5
        stressors = inp['blood_pressure_dev'] + inp['weight_deviation'] * .6 + inp['cancer_diagnosis'] * 30 + inp['age'] * .2 + inp['smoked_years'] * .8
        
        if inp['acute_infection_b'] and (not inp['prev_infection_b']) and (not inp['prev_infection_a']) and virtual_age > 45:
            return (self.CAT_DEATH, 'path1')
        
        if immune_overdrive > 12 and inp['blood_pressure_dev'] * 1.3 + virtual_age > 120:
            return (self.CAT_DEATH, 'path2')
        if immune_overdrive > 12 and inp['blood_pressure_dev'] + virtual_age > 80:
            return (self.CAT_HOSPITAL, 'path3')
        
        if stressors + virtual_age > 160:
            return (self.CAT_DEATH, 'path4')
        if stressors > 50 and virtual_age > 30:
            return (self.CAT_HOSPITAL, 'path5')
        
        if inp['prev_infection_a'] and (not inp['acute_infection_b']):
            return (self.CAT_HOME, 'path8')
        if inp['prev_infection_a'] and inp['acute_infection_b'] and inp['prev_infection_b'] and inp['age'] < 35:
            return (self.CAT_HOME, 'path9')
        if inp['prev_infection_a'] and inp['acute_infection_b'] and inp['age'] < 22:
            return (self.CAT_HOME, 'path10')
        
        if inp['acute_infection_b'] and virtual_age > 18:
            return (self.CAT_HOSPITAL, 'path11')
        
        if inp['age'] + stressors * .2 > 80:
            return (self.CAT_HOSPITAL, 'path12')

        return (self.CAT_HOME, 'path13')
        

def test_cat_distribution():
    """Test whether the three categories occur with equal probability"""
    o = GenerateData()
    stats = {o.CAT_DEATH: 0, o.CAT_HOSPITAL: 0, o.CAT_HOME: 0}
    for i in range(10000):
        r = o.rnd_inputs()
        c = o.input_to_category(r)
        stats[c[0]] += 1
    print(stats)
    # Output: {'CAT_DEATH': 3439, 'CAT_HOSPITAL': 3323, 'CAT_HOME': 3238}

def test_path_distribution():
    """To check the complexity of the toy example, test whether the individual paths occur with some useful frequency"""
    o = GenerateData()
    stats = {f"path{i}": 0 for i in range(1, 14)}
    for i in range(10000):
        r = o.rnd_inputs()
        c = o.input_to_category(r)
        stats[c[1]] += 1
    print(stats)

def test_inputs_mean_variance():
    """Calculate the mean and variance of input features"""
    samples = 1000000
    o = GenerateData()
    stats = {k: [] for k in o.rnd_inputs().keys()}
    for i in range(samples):
        for k, v in o.rnd_inputs().items():
            stats[k].append(v)
    for k, v in stats.items():
        v = np.array(v, dtype='float32')
        mean = np.mean(v)
        v -= mean
        var = np.var(v)
        print(f"{k}: var={var} mean={mean}")

def test_normalized_variance():
    """Test the mean/variance of the normalized input"""
    samples = 1000000
    o = GenerateData()
    stats = [[] for k in o.rnd_inputs().keys()]
    for i in range(samples):
        for ix, v in enumerate(o.normalize_inputs(o.rnd_inputs())):
            stats[ix].append(v)
    for ix, v in enumerate(stats):
        v = np.array(v, dtype='float32')
        mean = np.mean(v)
        v -= mean
        var = np.var(v)
        print(f"{ix}: var={var} mean={mean}")
        
def plot_outcomes(plot_paths=False):
    """Create a plot of outcomes along 2 variables"""
    o = GenerateData()
    display = {
        o.CAT_DEATH: '+',
        o.CAT_HOSPITAL: 'o',
        o.CAT_HOME: '.'
    }
    for age in range(0,100,5):
        for weight in range(-100,100,10):
            inp = {
                'prev_infection_a': 0,
                'prev_infection_b': 0,
                'acute_infection_b': 1,
                'cancer_diagnosis': 0, #1,
                'weight_deviation': weight,
                'age': age,
                'blood_pressure_dev': 0, #40,
                'smoked_years': 0, #5
            }
            c = o.input_to_category(inp)
            if plot_paths:
                c = c[1][-2:] + ","
            else:
                c = display[c[0]]
            print(c, end="")
        print("")
            
    

if __name__ == "__main__":
    
    test_cat_distribution()
    test_path_distribution()
    # test_inputs_mean_variance()
    # test_normalized_variance()
    
    plot_outcomes(plot_paths=False)

