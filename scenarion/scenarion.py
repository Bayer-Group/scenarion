import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


from sklearn.ensemble import IsolationForest


def test_scenarios(model, df, feature_columns, feat_to_test, n_scenarios=40, 
        classes_to_show='all', n_samples=100, id_col=None, show_plot=True,
        alpha='auto', hide_oos=False, oos_sensitivity='auto'):
    
    '''
    Generate scenarios (new samples) that differ in their values of col_to_test.
    Use model to make predictions for each of the generated scenarios.
    Return a DataFrame with the results of the scenario tests and create a figure
    visualizing those results.
    
    
    Parameters
    ----------
    model - sklearn-like model
        This should be a model that is already trained, and has a 
        model.predict_proba method and a model.classes_ attribute.
        
    df - pandas.DataFrame
        A dataframe that contains the data that you'd like to scenario test.
        It needs to have at least all columns in feature_columns.
        
    feature_columns - list of strings
        This is a list of column names, all of which must be found in df.
        This list must also correspond to the inputs/features that the model 
        was trained on (and be in the same order).
        
    feat_to_test - string or list of strings
        If  string, the name of the column to whose impact you'd like to explore 
        (a single string is used for a continuous feature).
        
        If a list of strings, each string should be a column name and each should
        correspond to an option for a one-hot-encoded categorical feature.
        
    n_scenarios - int
        The number of different values to test for each sample. These 
        values will be evenly distributed across the range of values found
        in the col_to_test column of df.
        Note: n_scenarios only has an effect when testing continuout features
        (because for categorical feature, scenarion will simply create a scenario
        for each possible value of the category).
        
    classes_to_show - string or list
        The only allowed string value is 'all' (default).
        If a list is passed, each element of the list must correspond to
        a value found in model.classes_
        Note: classes_to_show only has an effect when visualizing classification
        models.
        
    n_samples - string or int
        If int, the number of samples to use from df for performing the 
        scenario tests.
        The only allowed string value is 'all'. If 'all' is passed, all
        samples are used.
        
    id_col - string or None
        The name of the column in df to use as an identifier so that the
        results returned by this function can be reconnected with the
        original data. If None, the index of df will be used.
        
    show_plot - boolean
        Whether or not to create a figure illustrating the results.
        
    alpha - 'auto' or float
        The opacity (between 0 and 1.0) of the lines drawn if show_plot is
        True. If 'auto', the opacity will be set 10/n_samples.
        
    
    oos_sensitivity - 'auto' or float
        How aggressively the outlier detector should be to label a scenario
        as out-of-sample. If a float is used, it should be between 0 and 1.0.
        
    
    Returns
    -------
    A pandas DataFrame containing a row for every scenario tested for 
    every sample tested (ie, with n_samples x n_scenerios rows).
    
    The dataframe has the following columns:
        test_val - the value of col_to_test used for that scenario
        test_val_label - 
        old_index - an identifier of the sample tested. If id_col was
                given this column will be have that name rather than
                old_index. If id_col was not given, this will be the 
                index for the sample from the input df.
        *_prob - a probability for each class that the model can predict.
    
    '''

    # Check inputs    
    if not isinstance(df, pd.DataFrame):
        raise TypeError('df must be a pandas.DataFrame(), but was actually' +
                       ' a ' + str(type(df)))
        
    # Sample df to get the x data to use for building scenarios, and get the
    # corresponding IDs for those x data
    sampled_x, sampled_ids, output_id_col_name = do_sampling(df, feature_columns,
                                                            id_col, n_samples)
    
    if alpha == 'auto':
        alpha = np.clip(15/sampled_x.shape[0], 0.005, 1)
    
    # Figure out if the model type is classifier or regressor
    model_type = get_model_type(model)
        
    # Figure out if col_to_test is a continous variable or categorical
    feature_type = get_feature_type(feat_to_test, df)
    
    # Train the outlier detector to detect out-of-sample scenarios
    outlier_detector = IsolationForest(behaviour='new', max_samples=1.0,
                         contamination=oos_sensitivity)
    
    outlier_detector.fit(df[feature_columns].values)
    
        
    # Figure out the locations (indices) of the columns we'll be changing for 
    # each scenario
    if feature_type == 'continuous':
        min_val = df[feat_to_test].min()
        max_val = df[feat_to_test].max()
        
        test_col_loc = feature_columns.index(feat_to_test)
        
        test_vals = np.linspace(min_val, max_val, n_scenarios)
        
    elif feature_type == 'categorical':
        test_col_locs = [feature_columns.index(col) for col in feat_to_test]
        
    ####---------- DO THE SCENARIO TESTING -------------####
    
    # Create a dataframe to hold the results of our scenario tests
    temp_result_dfs = []
    
    for samp_id, sample in tqdm(zip(sampled_ids, sampled_x), desc='Testing Scenarios',
                               total=sampled_x.shape[0]):
        temp_sample = copy.copy(sample)
        
        # Get the new scenarios for this sample
        temp_df = pd.DataFrame()
        if feature_type == 'continuous':
            scenarios = np.vstack([replace_val(temp_sample, test_col_loc, test_val)
                            for test_val in test_vals])
            
            temp_df['test_val'] = test_vals
        
        elif feature_type == 'categorical':
            scenarios = np.vstack(build_categorical_scenarios(sample, test_col_locs))
            
            temp_df['test_val_label'] = feat_to_test
            temp_df['test_val'] = list(range(len(feat_to_test)))
        
        
        temp_df[output_id_col_name] = samp_id
        
        if model_type == 'classifier':
            results = model.predict_proba(scenarios)
            
            # Save the probabilities to our dataframe
            for i, class_ in enumerate(model.classes_):
                col_name = str(class_) + '_prob'
                temp_df[col_name] = results[:,i]
        
        elif model_type == 'regressor':
            results = model.predict(scenarios)
            
            # Save the predictions to our dataframe
            temp_df['prediction'] = results
        
        # Predict which scenarios are out-of-sample
        temp_df['in_sample'] = outlier_detector.predict(scenarios)
        
        temp_result_dfs.append(temp_df)
        
    results_df = pd.concat(temp_result_dfs)
    
    ####---------- CREATE THE PLOTS -------------####
    if show_plot:
        fig = plt.figure()
        
        if model_type == 'classifier': 
            # Make a subplot for each class to show
            class_prob_cols = [col for col in results_df.columns if
                               col.endswith('_prob')]

            if classes_to_show == 'all':
                classes_to_show = model.classes_

            n_classes_to_show = len(classes_to_show)

            # loop over each of the classes that we're showing
            # making a subplot for each one
            subplot_num = 1
            for class_ in classes_to_show:
                ax = fig.add_subplot(n_classes_to_show, 1, subplot_num)

                class_col_name = str(class_) + '_prob'

                # loop over each sample that was scenario tested
                # plotting a line for each
                for samp_id, sub_df in results_df.groupby(output_id_col_name):
                    insamp_predictions = np.ma.array(sub_df[class_col_name].values)
                    insamp_predictions = np.ma.masked_where(sub_df.in_sample < 0,
                                                       insamp_predictions)


                    ax.plot(sub_df.test_val, insamp_predictions, color='k',
                           alpha=alpha, label='_nolegend_')

                    if not hide_oos:
                        outsamp_predictions = np.ma.array(sub_df[class_col_name].values)
                        outsamp_predictions = np.ma.masked_where(sub_df.in_sample > 0,
                                                               outsamp_predictions)

                        ax.plot(sub_df.test_val, outsamp_predictions, color='r',
                               alpha=alpha)

                if not hide_oos:
                    ax.plot([np.nan, np.nan], label='Maybe out of sample', color='r')

                ax.legend()

                ax.set_ylabel(str(class_) + '\nProbability', rotation='horizontal',
                             horizontalalignment='right')
                
                if feature_type == 'categorical':
                    ax.set_xticks(list(range(len(feat_to_test))))
                    ax.set_xticklabels(feat_to_test, rotation=30, 
                                       horizontalalignment='right')
                
                elif feature_type == 'continuous':
                    ax.set_xlabel(feat_to_test)

                subplot_num += 1

            fig_height = len(classes_to_show) * 3
            fig.set_size_inches(10,fig_height)
            fig.subplots_adjust(hspace=0.3)
        
        elif model_type == 'regressor':
            ax = fig.add_subplot(111)
 
            # loop over each sample that was scenario tested
            # plotting a line for each
            for samp_id, sub_df in results_df.groupby(output_id_col_name):
                
                insamp_predictions = np.ma.array(sub_df.prediction.values)
                insamp_predictions = np.ma.masked_where(sub_df.in_sample < 0,
                                                       insamp_predictions)

                ax.plot(sub_df.test_val, insamp_predictions, color='k',
                       alpha=alpha, label='_nolegend_')

                if not hide_oos:
                    outsamp_predictions = np.ma.array(sub_df.prediction.values)
                    outsamp_predictions = np.ma.masked_where(sub_df.in_sample > 0,
                                                           outsamp_predictions)

                    ax.plot(sub_df.test_val, outsamp_predictions, color='r',
                           alpha=alpha)

            # Add a legend item for out-of-sample
            if not hide_oos:
                ax.plot([np.nan, np.nan], label='Maybe out of sample', color='r')

            ax.legend()

            ax.set_ylabel('Prediction', rotation='horizontal',
                         horizontalalignment='right')

            if feature_type == 'categorical':
                ax.set_xticks(list(range(len(feat_to_test))))
                ax.set_xticklabels(feat_to_test, rotation=30, 
                                       horizontalalignment='right')
                
            elif feature_type == 'continuous':
                ax.set_xlabel(feat_to_test)

            fig_height = 4
            fig.set_size_inches(10, fig_height)
            
        
    return results_df
    

def get_feature_type(col_to_test, df):
    if isinstance(col_to_test, str):
        if not col_to_test in df.columns:
            raise ValueError('col_to_test must be a column of df. Got ' + 
                            str(col_to_test) + 'for col_to_test')
            
        feature_type = 'continuous'
        
    elif isinstance(col_to_test, list):
        # make sure that each item in the list is a one-hot-encoded col
        for col in col_to_test:
            values_in_col = set(df[col].astype(int).unique())
            if not values_in_col <= set((0,1)):
                raise ValueError('If a list of columns is passed for ' +
                                'col_to_test, each of those columns ' + 
                                'must be a one-hot-encoded column ' +
                                ' but `' + str(col) + '` was in col_to_test' +
                                ' and is not one-hot-encoded. To test ' +
                                'scenarios for multiple continuous variables' +
                                ' call `test_scenarios` multiple times.')
        
        feature_type = 'categorical'
    
    return feature_type
    
    

def get_model_type(model):
    if hasattr(model, 'predict_proba'):
        model_type = 'classifier'
        
    elif hasattr(model, 'predict'):
        model_type = 'regressor'
        
    else:
        raise ValueError('model must have either a `.predict` or `.predict_proba` method.' +
        ' The object passed into model was a ' + str(type(model)) +
        '. Please check your inputs.')
        
    return model_type
    
        
def do_sampling(df, feature_columns, id_col, n_samples):
    x = df[feature_columns].values
    
    if id_col is None:
        ids = df.index
        output_id_col_name = 'old_index'
    else:
        ids = df[id_col].values
        output_id_col_name = id_col
    
    # Sample x and ids if appropriate
    if n_samples == 'all':
        sampled_x = x
        sampled_ids = ids
        
    elif isinstance(n_samples, int):
        # get a sample of x values
        sampled_indices = np.random.choice(x.shape[0], n_samples, replace=False)
        sampled_x = x[sampled_indices]
        
        # get the ids for the samples
        sampled_ids = ids[sampled_indices]
        
    else:
        raise ValueError("Got unexpected value for 'sample' argument: " + 
                         str(n_samples))
    
    
    return sampled_x, sampled_ids, output_id_col_name
    
        
def replace_val(a, location, new_val):
    '''
    Take an array and replace a single value within it.
    
    Parameters
    ----------
    a - array or list
        The array that you want to change.
        
    location - int
        The index/location of the element in `a` to replace.
        
    new_val
        The new value to put into the array.

    
    Return
    ------
    A copy of the array with the new value.
    
    '''
    new_a = copy.copy(a)
    new_a[location] = new_val
    return new_a


def build_categorical_scenarios(x, categorical_col_locs):
    '''
    Take a sample represented by an array, x, and generate several 
    new samples such that each generated sample differs only in the
    values found in categorical_col_locs. Specifically this function
    is meant to be used with one-hot-encoded categorical data.
    
    Example:
    
    [in]: build_categorical_scenarios([2,3,0,0,1], [2,3,4])
    [out]: [[2,3,1,0,0],
            [2,3,0,1,0],
            [2,3,0,0,1]]
    
    Parameters
    ----------
    x - array or list
        The sample array to use to generate scenarios from.
        
    categorical_col_locs - list of ints
        The positions in x that correspond to the one-hot-encoded
        categorical input.

    Return
    ------
    A list of arrays/lists containing the new scenarios
    '''
    scenarios = []
    
    for i in categorical_col_locs:
        new_x = copy.copy(x)
        for j in categorical_col_locs:
            if i == j:
                new_x[j] = 1
            else:
                new_x[j] = 0
                
        scenarios.append(new_x)
        
    return scenarios