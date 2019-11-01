from unittest import TestCase
import numpy as np
import pandas as pd

from scenarion.scenarion import (replace_val, build_categorical_scenarios)
import scenarion

class RegressorModel:
    def predict(self, x):
        return 3
        
        
class ClassifierModel:
    def __init__(self):
        self.classes_ = ['dog', 'cat']
        
    def predict_proba(self, x):
        return np.vstack([[0.8, 0.2] for _ in x])
        
    
class TestRegressor(TestCase):        
    def test_continuous(self):
        model = RegressorModel()
        
        df = pd.DataFrame()
        
        cont_1 = [1, 2, 3]
        cat_1 = [0, 1, 0]
        cat_2 = [1, 0, 1]
        
        df['cont'] = cont_1
        df['cat_1'] = cat_1
        df['cat_2'] = cat_2
        
        n_samples = 2
        n_scenarios = 4
        
        x_cols = ['cont', 'cat_1', 'cat_2']
        results = scenarion.test_scenarios(model, df, x_cols, 'cont', 
                                 show_plot=False, n_samples=n_samples, 
                                 n_scenarios=n_scenarios)
        
        print(results)
        
        self.assertTrue(isinstance(results, pd.DataFrame))
        self.assertIn('test_val', results.columns)
        self.assertIn('old_index', results.columns)
        self.assertIn('prediction', results.columns)
        self.assertEqual(len(results), n_scenarios*n_samples)
        
    
    def test_categorical(self):
        model = RegressorModel()
        
        df = pd.DataFrame()
        
        cont_1 = [1, 2, 3]
        cat_1 = [0, 1, 0]
        cat_2 = [1, 0, 1]
        
        df['cont'] = cont_1
        df['cat_1'] = cat_1
        df['cat_2'] = cat_2
        
        n_samples = 2
        
        x_cols = ['cont', 'cat_1', 'cat_2']
        results = scenarion.test_scenarios(model, df, x_cols, ['cat_1', 'cat_2'], 
                                 show_plot=False, n_samples=n_samples)
        
        self.assertTrue(isinstance(results, pd.DataFrame))
        self.assertIn('test_val', results.columns)
        self.assertIn('test_val_label', results.columns)
        self.assertIn('old_index', results.columns)
        self.assertIn('prediction', results.columns)
        self.assertEqual(len(results), 2*n_samples )
        
               
class TestClassifier(TestCase):
    def test_continuous(self):
        model = ClassifierModel()
        
        df = pd.DataFrame()
        
        cont_1 = [1, 2, 3]
        cat_1 = [0, 1, 0]
        cat_2 = [1, 0, 1]
        
        df['cont'] = cont_1
        df['cat_1'] = cat_1
        df['cat_2'] = cat_2
        
        n_samples = 2
        n_scenarios = 4
        
        x_cols = ['cont', 'cat_1', 'cat_2']
        results = scenarion.test_scenarios(model, df, x_cols, 'cont', 
                                 show_plot=False, n_samples=n_samples, 
                                 n_scenarios=n_scenarios)
        
        self.assertTrue(isinstance(results, pd.DataFrame))
        self.assertIn('test_val', results.columns)
        self.assertIn('old_index', results.columns)
        self.assertIn('dog_prob', results.columns)
        self.assertIn('cat_prob', results.columns)
        self.assertEqual(len(results), n_scenarios*n_samples)
        
    def test_categorical(self):
        model = ClassifierModel()
        
        df = pd.DataFrame()
        
        cont_1 = [1, 2, 3]
        cat_1 = [0, 1, 0]
        cat_2 = [1, 0, 1]
        
        df['cont'] = cont_1
        df['cat_1'] = cat_1
        df['cat_2'] = cat_2
        
        n_samples = 2
        
        x_cols = ['cont', 'cat_1', 'cat_2']
        results = scenarion.test_scenarios(model, df, x_cols, ['cat_1', 'cat_2'], 
                                 show_plot=False, n_samples=n_samples)
        
        self.assertTrue(isinstance(results, pd.DataFrame))
        self.assertIn('test_val', results.columns)
        self.assertIn('test_val_label', results.columns)
        self.assertIn('old_index', results.columns)
        self.assertIn('dog_prob', results.columns)
        self.assertIn('cat_prob', results.columns)
        self.assertEqual(len(results), 2*n_samples )
   
        
class TestReplaceVal(TestCase):
    def test_list(self):
        a = [1,2,3,4]
        location = 1
        new_val = 7
        result = [1,7,3,4]
        self.assertListEqual(replace_val(a, location, new_val),
                         result)
        
    def test_array(self):
        a = np.array([1,2,3,4])
        location = 1
        new_val = 7
        result = np.array([1,7,3,4])
        np.testing.assert_array_equal(replace_val(a, location, new_val), 
                         result)
        

class TestBuildCategoricalScenarios(TestCase):
    def test_list(self):
        x = [2,3,0,0,1]
        locations = [2,3,4]
        
        result = [[2,3,1,0,0],
                  [2,3,0,1,0],
                  [2,3,0,0,1]]
        
        self.assertListEqual(build_categorical_scenarios(x, locations),
                            result)
        
    def test_array(self):
        x = np.array([2,3,0,0,1])
        locations = [2,3,4]
        
        expected_results = [np.array([2,3,1,0,0]),
                  np.array([2,3,0,1,0]),
                  np.array([2,3,0,0,1])]
        
        for result, expected in zip(
                    build_categorical_scenarios(x, locations),
                    expected_results):
            
            np.testing.assert_array_equal(result, expected)