#reads attribute in .irap or .cps3 format and returns numpy array
def irap_cps3_reader(attribute_path):
    import numpy as np 
    
    tf = open(attribute_path, "r")
    line1 = tf.readline()
    
    #provided attributes are in one of two formats - .cps3 or .irap
    #those two formats are parsed in a different way
    f = "cps3" if "FSASCI" in line1 else "irap"
        
    if f == "cps3":
        skip, skip, skip, skip, xmin, xmax, ymin, ymax, skip, skip, skip, inldim, crldim, skip, xinc, yinc = np.array((tf.readline() + tf.readline()+ tf.readline() + tf.readline()).strip().replace('\n',' ').replace('  ',' ').split(' '))
        tf.readline()
        att = np.array(tf.read().strip().replace('\n',' ').replace('   ',' ').replace('  ',' ').split(' ')).astype(np.float).reshape((int(crldim),int(inldim)),order='C')
        att[att == 0.1E+31] = 9999900.0
        att = np.fliplr(att)
        
    else:
        skip,inldim,xinc,yinc,xmin,xmax,ymin,ymax,crldim,skip,skip,skip = np.array((line1+ tf.readline() + tf.readline()).strip().replace('\n',' ').replace('  ',' ').split(' ')).astype(np.float)
        tf.readline()
        att = np.array(tf.read().strip().replace('\n',' ').replace('   ',' ').replace('  ',' ').split(' ')).astype(np.float).reshape((int(crldim),int(inldim)),order='F')
        
    return att

#collects all necessary model parameters and puts them in one .json file
#all input arguments are defined by user in demo notebook
#definition of these five arguments is user's responsibility, all remaining parameters are created automatically
def build_config(training_data_path, test_data_path, number_of_configurations, predictions_folder_path):
        
    import json
    import os
    
    #set model parameters
    #use_pca is always false, as pca didn't give us better results
    #use_scaler is always true, and imputation strategy is always 'mean'
    #smoothing coefficient controls impact of linear model, can be seen as regularization parameter
    use_pca = False
    use_scaler = True
    smoothing_coefficient = 0.9
    save_feature_importance = True
    imputation_strategy = 'mean'
    
    #used only when use_pca = True, so never
    variance_explained_threshold = 0.999

    #id is solution's id, affects naming of all saved files
    #shuffled is always in training data path, and use_pca is always false, so this is redundant
    if use_pca == True:
        if "shuffled" in training_data_path:
            id = str(number_of_configurations // 1000) + "k_pca_new_val_set_linear"
        else:
            id = str(number_of_configurations // 1000) + "k_pca_linear"
    else:
        if "shuffled" in training_data_path:
            id = str(number_of_configurations // 1000) + "k_no_pca_new_val_set_linear"
        else:
            id = str(number_of_configurations // 1000) + "k_no_pca_linear"
    
    #config file is .json file containing all necessary parameters and paths where files should be saved
    config_path = "config_files/config_" + str(id) + ".json"
    
    #all XGB configurations and optimal configurations are saved in .csv format, all optimal models are saved to .pkl and predictions on training
    #and test set are also saved in .csv
    model_configurations_path = "model_configurations/all_configurations_" + id + ".csv"
    optimal_model_configurations_path = "model_configurations/optimal_configurations_" + id + ".csv"
    optimal_models_path = "optimal_models/" + id + "/"
    train_predictions_path = predictions_folder_path + "train_predictions_" + id + ".csv"
    test_predictions_path = predictions_folder_path + "test_predictions_" + id + ".csv"
    #this dictionary contains key informations regarding training process (number of optimal configurations and metrics)
    dict_path = predictions_folder_path + "mse_dictionary_" + id + ".txt"

    #attribute feature importance is also saved to a .csv file
    feature_importance_path = "feature_importance/mean_feature_importance_" + id + ".csv"

    #visualizations (optimal models cross plots and NTG histograms) are stored in one folder
    cross_plots_path = "plots/cross_plots/" + id + "/"
    ntg_hist_path = "plots/ntg_histograms/" + id

    #standard scaler and pca transformer are saved to pickle files
    scaler_path = "scaler.pkl"
    pca_path = "pca" + str(variance_explained_threshold) + ".pkl"

    #correlation matrices for training and test set are saved to .csv
    #not used in current solution, redundant
    training_corr_path = "correlation_matrix_training_data.csv"
    test_corr_path = "correlation_matrix_test_data.csv"
    
    #predictions are converted from .csv to .irap format, so they can be visualized easily
    #maps are just intermediate step, they are used just to come to .irap format
    #this step can probably be skipped, somehow
    prediction_map_path_mean = predictions_folder_path + "Hef_prediction_map_" + id + ".txt"
    prediction_map_path_p10 = predictions_folder_path + "Hef_prediction_p10_map_" + id + ".txt"
    prediction_map_path_p50 = predictions_folder_path + "Hef_prediction_p50_map_" + id + ".txt"
    prediction_map_path_p90 = predictions_folder_path + "Hef_prediction_p90_map_" + id + ".txt"
    prediction_irap_path_mean = predictions_folder_path + "Hef_prediction_irap_" + id + ".txt"
    prediction_irap_path_p10 = predictions_folder_path + "Hef_prediction_p10_irap_" + id + ".txt"
    prediction_irap_path_p50 = predictions_folder_path + "Hef_prediction_p50_irap_" + id + ".txt"
    prediction_irap_path_p90 = predictions_folder_path + "Hef_prediction_p90_irap_" + id + ".txt"
    
    #indicates if values in test attributes are missing
    #used to filter attributes before their merging, in order to avoid huge datasets
    indices_path = "indices.npy"
    
    config_dict = {'config_path': config_path,
                   'number_of_configurations': number_of_configurations,
                   'use_pca': use_pca,
                   'model_configurations_path': model_configurations_path,
                   'optimal_model_configurations_path': optimal_model_configurations_path,
                   'optimal_models_path': optimal_models_path,
                   'train_predictions_path': train_predictions_path,
                   'test_predictions_path': test_predictions_path,
                   'dict_path': dict_path,
                   'feature_importance_path': feature_importance_path,
                   'cross_plots_path': cross_plots_path,
                   'ntg_hist_path': ntg_hist_path,
                   'save_feature_importance': save_feature_importance,
                   'imputation_strategy': imputation_strategy,
                   'use_scaler': use_scaler,
                   'scaler_path': scaler_path,
                   'pca_path': pca_path,
                   'training_corr_path': training_corr_path,
                   'test_corr_path': test_corr_path,
                   'smoothing_coefficient': smoothing_coefficient,
                   'prediction_map_path_mean': prediction_map_path_mean,
                   'prediction_map_path_p10': prediction_map_path_p10,
                   'prediction_map_path_p50': prediction_map_path_p50,
                   'prediction_map_path_p90': prediction_map_path_p90,
                   'prediction_irap_path_mean': prediction_irap_path_mean,
                   'prediction_irap_path_p10': prediction_irap_path_p10,
                   'prediction_irap_path_p50': prediction_irap_path_p50,
                   'prediction_irap_path_p90': prediction_irap_path_p90,
                   'indices_path': indices_path
                   }
    
    #create folder if needed and save config file to .json
    if not os.path.exists("config_files/"):
        os.makedirs("config_files/", exist_ok = True)
        
    with open(config_path, 'w') as file:
         file.write(json.dumps(config_dict))
    
    return config_dict

#based on model parameters, performs some or all of the following steps: missing value imputation, standard scaling and pca transformation
#returns results of these transformations
#could take config as only argument, since all other parameters are contained in config, don't know why it's like this
def data_transform(train_data,
                   use_scaler,
                   use_pca,
                   imputation_strategy,
                   scaler_path,
                   pca_path):
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    
    from pickle import dump
    from pickle import load
    
    import numpy as np
    import pandas as pd
    import os
    
    #imputation is always performed, and it's always mean
    imp = SimpleImputer(missing_values=np.nan, strategy=imputation_strategy)
    imp.fit(train_data)
    train_data_imputed = pd.DataFrame(imp.transform(train_data), columns = train_data.columns)
    
    #train and test split of 22 points dataset is performed by simple slicing
    #not a problem, since dataset was initially shuffled
    x = train_data_imputed.loc[:,train_data_imputed.columns != "Hef_Bd"].iloc[:16]
    y = train_data_imputed.Hef_Bd.iloc[:16]
    x_val = train_data_imputed.loc[:,train_data_imputed.columns != "Hef_Bd"].iloc[16:]
    y_val = train_data_imputed.Hef_Bd.iloc[16:] 
    
    #used only when use_pca is True, so never
    variance_explained_threshold = 0.999
    
    #if scaler already exists, just load it, otherwise, fit a standard scaler to training data
    if use_scaler == True:
        if os.path.exists(scaler_path):
            scaler = load(open(scaler_path, 'rb'))
        else:
            scaler = StandardScaler()
            scaler.fit(x)

        x_scaled = scaler.transform(x)
        x_val_scaled = scaler.transform(x_val)
        
        dump(scaler, open(scaler_path, 'wb'))
    
    #same goes for pca
    if use_pca == True:
        if os.path.exists(pca_path):
            pca = load(open(pca_path, 'rb'))
        else:
            pca = PCA(n_components = variance_explained_threshold)
            pca.fit(x_scaled)
        
        x_pca_scaled = pca.transform(x_scaled)
        x_val_pca_scaled = pca.transform(x_val_scaled)
            
        dump(pca, open(pca_path, 'wb'))
        
        return x_scaled, x_val_scaled, x_pca_scaled, x_val_pca_scaled, y, y_val
    
    else:
        #returns tuple either way, in this case (which is default case, at the moment) results of scaling is returned
        return x_scaled, x_val_scaled, y, y_val    
    
#saves attribute correlations in two .csv files (for training and test attributes) and plots correlation matrices
#due to non-informative correlation plots, not used in current solution
def save_correlation_matrices(training_data_path,
                              test_data_path,
                              conf):
    import pandas as pd
    
    def plot_correlation_matrix(df, title):
        from matplotlib import pyplot as plt
        plt.clf()
        plt.figure(figsize = (30,30))
        plt.imshow(df.corr(), interpolation = 'nearest')
        plt.xticks(range(df.shape[1]), df.columns, fontsize = 8, rotation = 45)
        plt.yticks(range(df.shape[1]), df.columns, fontsize = 8)
        plt.colorbar()
        plt.title(title, fontsize = 16)
        plt.show()
    
    training_corr_path = conf['training_corr_path']
    test_corr_path = conf['test_corr_path']
                              
    train_data = pd.read_csv(training_data_path, index_col = 0)
    test_data = pd.read_csv(test_data_path, index_col = 0)
    
    train_data.corr().to_csv(training_corr_path)
    test_data.corr().to_csv(test_corr_path)
    
    plot_correlation_matrix(train_data, "Training Data Correlation Matrix")
    plot_correlation_matrix(test_data, "Test Data Correlation Matrix")

#auxilliary function for R^2 metric calculation
def r_squared(y, y_pred):
    import numpy as np
    y_mean = np.mean(y)
    ss_total = np.sum((y - y_mean) ** 2)
    ss_res = np.sum((y - y_pred) ** 2)
         
    return 1 - ss_res / ss_total

#auxilliary function for deletion of all files in a given folder
def empty_folder(folder_path):
    import os
    for filename in os.listdir(folder_path):
        os.remove(folder_path + filename)
        
#function creates given number of XGB regressor configurations, trained on given training data
#output is a dataframe where each row is one configuration, associated with its metrics on training and validation data
def create_configurations(training_data_path, 
                          conf):
    
    import pandas as pd 
    import random
    import xgboost
    import numpy as np
    import os
    from IPython.display import clear_output
    
    data = pd.read_csv(training_data_path, index_col = 0)
    
    #take necessary parameters from config file
    number_of_configurations = conf['number_of_configurations']
    use_pca = conf['use_pca']
    configurations_path = conf['model_configurations_path']
    use_scaler = conf['use_scaler']
    scaler_path = conf['scaler_path']
    pca_path = conf['pca_path']
    imputation_strategy = conf['imputation_strategy']
    
    #first remove rows with missing target value, and perform data transformation, using data_transform function
    train_data = data[data["Hef_Bd"].notnull()]
    if use_pca == False:
        x_scaled, x_val_scaled, y, y_val = data_transform(train_data, 
                                                          use_scaler = use_scaler, 
                                                          use_pca = use_pca, 
                                                          imputation_strategy = imputation_strategy,
                                                          scaler_path = scaler_path,
                                                          pca_path = pca_path)
    else:
        _, _, x_scaled, x_val_scaled, y, y_val = data_transform(train_data, 
                                                          use_scaler = use_scaler, 
                                                          use_pca = use_pca, 
                                                          imputation_strategy = imputation_strategy,
                                                          scaler_path = scaler_path,
                                                          pca_path = pca_path)
    
    #this list will gather information about all created configurations, and will be transformed to a dataframe at the end
    configurations = []
    for i in range(number_of_configurations):
        
        #randomly generate XGBRegressor hyperparameters
        #hyperparameters are chosen so that overfitting is discouraged, by using shallower trees and smaller number of trees
        colsample_by_tree = (np.random.randint(4) + 1) / 4
        learning_rate = random.uniform(0.01,2)
        max_depth = np.random.randint(10) + 1
        n_estimators = random.choice([1,2,3,4,5,6,7,8,9,10,12,15,20,25,30,35,40,45,50])
        subsample = (np.random.randint(4) + 1) / 4
        reg_alpha = random.uniform(0.01,1)
        reg_lambda = random.uniform(0.01,1)
    
        #model is trained and fit on training data
        model = xgboost.sklearn.XGBRegressor(colsample_by_tree = colsample_by_tree,
                                             learning_rate = learning_rate,
                                             max_depth = max_depth,
                                             n_estimators = n_estimators,
                                             subsample = subsample,
                                             reg_alpha = reg_alpha,
                                             reg_lambda = reg_lambda,
                                             verbosity = 0)
        
        model.fit(X = x_scaled, y = y)
        
        #predictions are calculated for training, validation and training + validation set
        train_predictions = model.predict(x_scaled)
        val_predictions = model.predict(x_val_scaled)
        predictions = np.concatenate((train_predictions, val_predictions))
        y_all = np.concatenate((y,y_val))
        
        #MSE and R^2 are calculated for training, validation and training + validation set
        train_mse = np.mean((train_predictions - y) ** 2)
        val_mse = np.mean((val_predictions - y_val) ** 2)
        total_mse = np.mean((predictions - y_all) ** 2)
        
        train_r2 = r_squared(y, train_predictions)
        val_r2 = r_squared(y_val, val_predictions)
        total_r2 = r_squared(y_all, predictions)
        
        #a list containing values of hyperparameters and corresponding model's metrics is added to the configuration list of lists
        configurations.append([colsample_by_tree, learning_rate, max_depth, n_estimators, 
                               subsample, reg_alpha, reg_lambda, 
                               train_mse, val_mse, total_mse,
                               train_r2, val_r2, total_r2])
        
        #used for some kind of progress bar in the demo notebook
        clear_output(wait=True)
        print(f"{i+1} / {number_of_configurations} model configurations created, please wait...", flush=True)
    
    #message indicating end of model generation
    print(f"{number_of_configurations} model configurations created.", flush = True)
    #transforming to dataframe and saving configurations info to .csv
    configurations_df = pd.DataFrame(configurations, columns = ["colsample_by_tree","learning_rate","max_depth",
                                                                "n_estimators","subsample","reg_alpha","reg_lambda",
                                                                "train_mse","val_mse","total_mse",
                                                                "train_r2", "val_r2", "total_r2"])
    
    if not os.path.exists("model_configurations/"):
        os.makedirs("model_configurations/", exist_ok = True)
    
    configurations_df.to_csv(configurations_path)

#function creates linear regression model for smoothing 
#output is model itself, and width of 80%-confidence interval, whose limits will be used for smoothing p10/p90 maps
def create_linear_model(training_data_path,
                        conf):
    import pandas as pd
    import numpy as np

    from sklearn import linear_model
    from sklearn.impute import SimpleImputer
    from pickle import load
    
    #function calculates width of confidence interval, for given true and predicted values of target variable
    #80%-confidence interval is chosen, so that its limits can be used for smoothing p10 and p90 maps
    def calculate_confidence_interval(y_test, y_predicted, p = .8):
        
        import numpy as np
        import scipy.stats
        sum_errs = np.sum((y_test - y_predicted) ** 2)
        stdev = np.sqrt(1 / (len(y_test) - 2) * sum_errs)
        
        one_minus_p = 1 - p
        ppf_lookup = 1 - (one_minus_p / 2)
        z_score = scipy.stats.norm.ppf(ppf_lookup)
        
        interval = z_score * stdev
        
        return interval
    
    #data is loaded and transformed
    #not sure why data_transform is not used here
    df = pd.read_csv(training_data_path, index_col = 0)
    
    df_notna = df[df["Hef_Bd"].notnull()]
    
    X = df_notna.loc[:, df.columns != "Hef_Bd"]
    Y = df_notna["Hef_Bd"]
    
    imputation_strategy = conf['imputation_strategy']
    scaler_path = conf['scaler_path']
    pca_path = conf['pca_path']
    use_pca = conf['use_pca']
    use_scaler = conf['use_scaler']
    
    imp = SimpleImputer(missing_values=np.nan, strategy=imputation_strategy)
    imp.fit(X)
    X_imputed = pd.DataFrame(imp.transform(X), columns = X.columns)
    
    X_train, X_val, Y_train, Y_val = X_imputed[:16], X_imputed[16:], Y[:16], Y[16:]
    if use_scaler:
        scaler = load(open(scaler_path, 'rb'))
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_scaled = scaler.transform(X_imputed)
    if use_pca:
        pca = load(open(pca_path, 'rb'))
        X_train_scaled = pca.transform(X_train_scaled)
        X_val_scaled = pca.transform(X_val_scaled)
        X_scaled = pca.transform(X_scaled)
    
    #create a linear model, then use it to calculate width of confidence interval
    linreg = linear_model.LinearRegression()
    linreg.fit(X_train_scaled, Y_train)
    Y_pred = linreg.predict(X_scaled)
    
    interval_width = calculate_confidence_interval(Y, Y_pred, 0.8)
    
    #return model and width
    return linreg, interval_width

#function applies user-created rules for selection of optimal configurations
#XGB aggregated predictions are smoothed using previously created linear model
#function provides prediction for training and validation set, one dictionary containing number of optimal configurations and metrics of interest
#some interesting interpretation tools are also provided, as cross plots for all optimal models and aggregated model, or feature importance matrix which will be analyzed later
def select_optimal_configurations(training_data_path,
                                  conf):
    
    import pandas as pd
    import xgboost
    import numpy as np
    from pickle import dump
    import json
    import os
    
    #first, take all needed parameters from config file
    configurations_path = conf['model_configurations_path']
    optimal_configurations_path = conf['optimal_model_configurations_path']
    optimal_models_path = conf['optimal_models_path']
    train_predictions_path = conf['train_predictions_path']
    visualization_path = conf['cross_plots_path']
    feature_importance_path = conf['feature_importance_path']
    use_scaler = conf['use_scaler']
    use_pca = conf['use_pca']
    imputation_strategy = conf['imputation_strategy']
    save_feature_importance = conf['save_feature_importance']
    scaler_path = conf['scaler_path']
    pca_path = conf['pca_path']
    dict_path = conf['dict_path']
    alpha = conf['smoothing_coefficient']

    #read all model configurations and matching metrics, saved by create_configurations() function
    #then, apply designed rules for selecting what we call "optimal" configurations
    #those filtered configurations are saved to another .csv
    all_configurations = pd.read_csv(configurations_path, index_col = 0)
    optimal_configurations = all_configurations.loc[all_configurations.val_mse < 50][all_configurations.train_mse < 50][all_configurations.train_mse > all_configurations.val_mse / 3].sort_values(by = "val_mse").iloc[:100,]
    optimal_configurations.to_csv(optimal_configurations_path)
    
    #linear model and confidence interval width are calculated
    linreg, confidence_interval_width = create_linear_model(training_data_path = training_data_path,
                                                            conf = conf)
    
    #function plots cross plot and saves it to a given path
    #column1 should be true values of target variable, and column2 should be predicted values
    def cross_plot(column1, column2, path):
        import numpy as np
        from matplotlib import pyplot as plt
    
        X_plot = np.linspace(0, 50, 50)
        Y_plot = np.linspace(0, 50, 50)
    
        plt.clf()
    
        plt.scatter(column1[:16], column2[:16], c = 'k', label = "train dataset")
        plt.scatter(column1[16:], column2[16:], c = 'b', label = "validation dataset")
        plt.legend()
        plt.plot(X_plot, Y_plot, c = "red", linestyle = "solid")
        plt.xlabel("Hef_Bd")
        plt.ylabel("prediction")
    
        plt.savefig(path)
    
    #prediction table is where training set predictions will be saved
    #we simply load the training data, and then we'll add prediction columns
    prediction_table = pd.read_csv(training_data_path, index_col = 0)
    prediction_table = prediction_table[prediction_table.Hef_Bd.notnull()]
    
    #transforming training data, so we use it to generate optimal models predictions
    if use_pca == False:    
        x_scaled, x_val_scaled, y, y_val = data_transform(prediction_table, 
                                                          use_scaler = use_scaler, 
                                                          use_pca = use_pca, 
                                                          imputation_strategy = imputation_strategy,
                                                          scaler_path = scaler_path,
                                                          pca_path = pca_path)
    else:
        x_scaler_only, x_val_scaler_only, x_scaled, x_val_scaled, y, y_val = data_transform(prediction_table, 
                                                          use_scaler = use_scaler, 
                                                          use_pca = use_pca, 
                                                          imputation_strategy = imputation_strategy,
                                                          scaler_path = scaler_path,
                                                          pca_path = pca_path)
    
    number_of_configurations = optimal_configurations.shape[0]
    #final predictions will be average of optimal models predictions, so we set it to zero, and then we'll sum it up
    final_predictions = np.zeros(prediction_table.shape[0])
    
    #attribute columns are needed, just for indexing of feature importance table
    if use_pca == False:
        attribute_column_names = prediction_table.columns[prediction_table.columns != "Hef_Bd"]
    else:
        attribute_column_names = list()
        for i in range(x_scaled.shape[1]):
            attribute_column_names.append("PC" + str(i + 1))
            
    #feature importance table will also be calculated as average of optimal models feature importance, so we again initialize with zeroes
    if save_feature_importance == True:
        final_feature_weights = pd.DataFrame(np.zeros(x_scaled.shape[1]), columns=['weights'], index = attribute_column_names)
        
    if not os.path.exists(optimal_models_path):
        os.makedirs(optimal_models_path, exist_ok = True)
    
    if not os.path.exists(visualization_path):
        os.makedirs(visualization_path, exist_ok = True)
        
    empty_folder(optimal_models_path)
    empty_folder(visualization_path)
        
    print("Selecting optimal model configurations...")
        
    for i in range(number_of_configurations):
        
        #we take the hyperparameters of each optimal configuration, and train new model with same parameters
        #another option is to save all created configurations to .pkl, and then only load optimal models
        #since training optimal models doesn't take too long, this solution is fast enough
        row = optimal_configurations.iloc[i,:]
        model = xgboost.sklearn.XGBRegressor(colsample_by_tree = row.colsample_by_tree,
                                             learning_rate = row.learning_rate,
                                             max_depth = int(row.max_depth),
                                             n_estimators = int(row.n_estimators),
                                             subsample = row.subsample,
                                             reg_alpha = row.reg_alpha,
                                             reg_lambda = row.reg_lambda)
        model.fit(X = x_scaled, y = y)
        
        dump(model, open(optimal_models_path + "model" + str(i) + ".pkl", 'wb'))
        
        #add each optimal model's feature importance to the feature importance table, will be averaged later
        if save_feature_importance == True:
            new_weights = pd.DataFrame(model.feature_importances_, columns=['weights'], index = attribute_column_names)
            final_feature_weights = final_feature_weights + new_weights
            
        #generate predictions for training, validation and training + validation set
        train_predictions = model.predict(x_scaled)
        val_predictions = model.predict(x_val_scaled)
        predictions = np.clip(np.append(train_predictions, val_predictions), 0, prediction_table.Htot_Bd)
        #clipping is performed to obtain logical results, as effective thickness must be in [0, total thickness] interval    
        
        #add each optimal model's prediction to the final prediction, will be averaged later and save each optimal model's prediction as separate column
        final_predictions += predictions 
        
        column_name = "prediction_model" + str(i)
        prediction_table[column_name] = predictions
        
        #save each optimal model's cross plot to designated path
        cross_plot(prediction_table["Hef_Bd"], prediction_table[column_name], visualization_path + "plot_model" + str(i) + ".png")
        
    #generate linear predictions for training, validation and training+validation set (clipping is again necessary)
    linear_train_prediction = linreg.predict(x_scaled)
    linear_val_prediction = linreg.predict(x_val_scaled)
    
    linear_prediction = np.clip(np.append(linear_train_prediction, linear_val_prediction), 0, prediction_table.Htot_Bd)
    #dividing with number of optimal configurations gives average optimal prediction
    xgb_prediction = final_predictions / number_of_configurations
    
    #correcting xgb prediction with weighted sum, using linear model prediction
    prediction_table['final_prediction'] = alpha * xgb_prediction + (1 - alpha) * linear_prediction
    #ntg is simply effective thickness / total thickness
    prediction_table['ntg_prediction'] = prediction_table.final_prediction / prediction_table.Htot_Bd

    #take all optimal model prediction columns as one sample, and use it to generate p10, p50 and p90 predictions
    prediction_columns = [col for col in prediction_table if col.startswith('prediction_model')]
    
    #note that p90 prediction is associated with 0.1-percentile, due to some weird business request that percentiles should be upside-down
    prediction_table['prediction_p90'] = alpha * prediction_table[prediction_columns].quantile(q = 0.1, axis = 1) + (1 - alpha) * (linear_prediction - confidence_interval_width)
    prediction_table['prediction_p50'] = alpha * prediction_table[prediction_columns].quantile(q = 0.5, axis = 1) + (1 - alpha) * linear_prediction
    prediction_table['prediction_p10'] = alpha * prediction_table[prediction_columns].quantile(q = 0.9, axis = 1) + (1 - alpha) * (linear_prediction + confidence_interval_width)
    
    print(f"{number_of_configurations} optimal model configurations selected.")
    
    #cross plot of mean, p10, p50 and p90 optimal prediction
    cross_plot(prediction_table["Hef_Bd"], prediction_table["final_prediction"], visualization_path + "plot_mean_model" + ".png")
    cross_plot(prediction_table["Hef_Bd"], prediction_table["prediction_p90"], visualization_path + "plot_p90_model" + ".png")
    cross_plot(prediction_table["Hef_Bd"], prediction_table["prediction_p50"], visualization_path + "plot_p50_model" + ".png")
    cross_plot(prediction_table["Hef_Bd"], prediction_table["prediction_p10"], visualization_path + "plot_p10_model" + ".png")
    
    #save prediction dataframe to train_predictions_path
    prediction_table.to_csv(train_predictions_path)
    
    #calculate mse and R^2 for mean model on training, validation and training+validation set
    train_mse = np.mean((prediction_table['Hef_Bd'][:16] - prediction_table['final_prediction'][:16]) ** 2)
    val_mse = np.mean((prediction_table['Hef_Bd'][16:] - prediction_table['final_prediction'][16:]) ** 2)
    total_mse = np.mean((prediction_table['Hef_Bd'] - prediction_table['final_prediction']) ** 2)
    
    train_r2 = r_squared(prediction_table['Hef_Bd'][:16],prediction_table['final_prediction'][:16])
    val_r2 = r_squared(prediction_table['Hef_Bd'][16:],prediction_table['final_prediction'][16:])
    total_r2 = r_squared(prediction_table['Hef_Bd'],prediction_table['final_prediction'])
    
    #dividing feature importance weight with number of optimal configurations gives average feature importance for each attribute
    #this feature importance table is sort, starting from most important attribute, and saved to a .csv file
    if save_feature_importance == True:
        if not os.path.exists("feature_importance/"):
            os.makedirs("feature_importance/", exist_ok = True)
        
        final_feature_weights["weights"] = final_feature_weights["weights"] / number_of_configurations
        final_feature_weights.sort_values("weights", ascending = 0).to_csv(feature_importance_path)
        
    #one dictionary containing training process data is saved to .txt file
    d = {'number_of_optimal_configurations': number_of_configurations,
            'train_set_mse': train_mse,
            'validation_set_mse': val_mse,
            'total_mse': total_mse,
            'train_set_r_squared': train_r2,
            'validation_set_r_squared': val_r2,
            'total_r_squared': total_r2}
    
    
    
    with open(dict_path, 'w') as file:
         file.write(json.dumps(d))
            
    return d

#auxilliary function which prints results of previous function in a fairly fashionable way
def print_training_results(a):
    if a['number_of_optimal_configurations'] > 0:
        #print("Number of optimal configurations is " + str(a['number_of_optimal_configurations']) + ".")
        print(f"Mean squared error on training set is {a['train_set_mse']}. Mean squared error on validation set is {a['validation_set_mse']}.")
        print(f"R-squared coefficient on training set is {a['train_set_r_squared']}. R-squared coefficient on validation set is {a['validation_set_r_squared']}.")
        print(f"Total mean squared error is {a['total_mse']}. Total R-squared coefficient is {a['total_r_squared']}.")
    else:
        print('No optimal configurations. Please increase number of configurations!')
  
#function takes feature importance table and plots it via pie plot    
def plot_feature_importance(conf):
    import pandas as pd
    from matplotlib import pyplot as plt
    import numpy as np
    df = pd.read_csv(conf['feature_importance_path'], names = ['attribute', 'weight'], header = 0)[:8]    
    labels = np.append(np.array(df.attribute), 'Other')
    sizes = np.append(np.array(df.weight), 1 - np.sum(df.weight))

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes,  labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()  
    
    return 

#functiontakes feature importance table, clusters attributes by their group, and prints pie plot of grouped importance
def plot_grouped_feature_importance(conf):
    import pandas as pd
    from matplotlib import pyplot as plt
    import numpy as np

    df = pd.read_csv(conf['feature_importance_path'], names = ['attribute', 'weight'], header = 0)

    average_energy_weights = np.sum(df[df.attribute.str.contains('Average energy')]['weight'])
    extract_value_weights = np.sum(df[df.attribute.str.contains('Extract value')]['weight'])
    mean_amplitude_weights = np.sum(df[df.attribute.str.contains('Mean amplitude')]['weight'])
    sum_of_amplitudes_weights = np.sum(df[df.attribute.str.contains('Sum of amplitudes')]['weight'])
    median_amplitude_weights = np.sum(df[df.attribute.str.contains('Median')]['weight'])
    minimum_amplitude_weights = np.sum(df[df.attribute.str.contains('Minimum amplitude')]['weight'])
    maximum_amplitude_weights = np.sum(df[df.attribute.str.contains('Maximum amplitude')]['weight'])
    rms_amplitude_weights = np.sum(df[df.attribute.str.contains('RMS amplitude')]['weight'])
    h_tot_weights = np.sum(df[df.attribute.str.contains('Htot')]['weight'])

    labels = ['RMS amplitude', 'Htot', 'Median amplitude', 'Mean amplitude', 'Maximum amplitude',
          'Other']
    
    sizes = [rms_amplitude_weights, h_tot_weights, median_amplitude_weights, mean_amplitude_weights,
             maximum_amplitude_weights, minimum_amplitude_weights + extract_value_weights + average_energy_weights + sum_of_amplitudes_weights]



    fig1, ax1 = plt.subplots()
    ax1.pie(sizes,  labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()  
    
    return

#function takes previously created ensemble model, and uses it for prediction on unlabeled data
#.csv file containing prediction is provided as a result
def predict_on_test_data(training_data_path,
                         test_data_path,
                         conf):
    
    import os
    import pandas as pd
    import numpy as np
    from sklearn.impute import SimpleImputer
    from pickle import load
    
    #read test data and take all necessary parameters from config file
    test_data = pd.read_csv(test_data_path, index_col=0)
    
    use_pca = conf['use_pca']
    optimal_models_path = conf['optimal_models_path']
    test_predictions_path = conf['test_predictions_path']
    use_scaler = conf['use_scaler']
    imputation_strategy = conf['imputation_strategy']
    scaler_path = conf['scaler_path']
    pca_path = conf['pca_path']
    
    #get linear model and its confidence interval
    linreg, confidence_interval_width = create_linear_model(training_data_path = training_data_path,
                                                            conf = conf)
    
    #again, not sure why data_transform is not used
    imp = SimpleImputer(missing_values=np.nan, strategy=imputation_strategy)
    imp.fit(test_data)
    test_data_imputed = pd.DataFrame(imp.transform(test_data), columns = test_data.columns)
    
    if use_scaler:
        scaler = load(open(scaler_path, 'rb'))
        test_data_scaled = scaler.transform(test_data_imputed)
        
    if use_pca:
        pca = load(open(pca_path, 'rb'))
        test_data_scaled = pca.transform(test_data_scaled)
        
    optimal_model_names = os.listdir(optimal_models_path)
    
    hef_predicted = np.zeros(test_data.shape[0])
    
    #load each optimal model, ensure that column order is not messed up, and generate each optimal model's prediction on test data
    #similarly as before, clipping is used
    for model_name in optimal_model_names:
        
        model = load(open(optimal_models_path + model_name, 'rb'))
        
        cols_when_model_builds = model.get_booster().feature_names

        pred = model.predict(pd.DataFrame(test_data_scaled, columns = cols_when_model_builds))
        column_name = "prediction_" + model_name
        test_data[column_name] = np.clip(pred, 0, test_data["Htot_Bd"])
        hef_predicted = hef_predicted + test_data[column_name]
     
    #get linear prediction for weighted sum
    linear_prediction = np.clip(linreg.predict(pd.DataFrame(test_data_scaled, columns = cols_when_model_builds)), 0, test_data["Htot_Bd"])
    
    
    #just like before, final xgb prediction is corrected with linear prediction, NTG and p10, p50, p90 are calculated
    xgb_prediction = hef_predicted / len(optimal_model_names)
    alpha = conf['smoothing_coefficient']
    
    test_data["Hef_predicted"] = alpha * xgb_prediction + (1 - alpha) * linear_prediction
    
    test_data["NTG_predicted"] = test_data["Hef_predicted"] / (test_data["Htot_Bd"] + 0.0000001)
    
    prediction_columns = [col for col in test_data if col.startswith('prediction_model')]
    test_data['Hef_predicted_p90'] = np.clip(alpha * test_data[prediction_columns].quantile(q = 0.1, axis = 1) + (1 - alpha) * (linear_prediction - confidence_interval_width), 0, test_data["Htot_Bd"])
    test_data['Hef_predicted_p50'] = np.clip(alpha * test_data[prediction_columns].quantile(q = 0.5, axis = 1) + (1 - alpha) * linear_prediction, 0, test_data["Htot_Bd"])
    test_data['Hef_predicted_p10'] = np.clip(alpha * test_data[prediction_columns].quantile(q = 0.9, axis = 1) + (1 - alpha) * (linear_prediction + confidence_interval_width), 0, test_data["Htot_Bd"])
    
    #test data with predictions is saved
    test_data.to_csv(test_predictions_path)
    
#function takes test set prediction and provides histogram of predicted effective thickness and NTG 
def draw_hef_ntg_distributions(conf):
    
    import pandas as pd
    from matplotlib import pyplot as plt
    import numpy as np
    import os
    
    test_data = pd.read_csv(conf['test_predictions_path'])
    ntg_hist_path = conf['ntg_hist_path']
        
    if not os.path.exists(ntg_hist_path):
        os.makedirs(ntg_hist_path, exist_ok = True)
    
    plt.clf()
    f, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (20,5))
    import scipy.stats as st
    ax1.hist(test_data["NTG_predicted"], density=True, stacked = True, bins=30, label="NTG prediction")

    kde_xs = np.linspace(0, 1, 301)
    kde = st.gaussian_kde(test_data["NTG_predicted"])
    ax1.plot(kde_xs, kde.pdf(kde_xs), label="Density function")
    ax1.legend(loc="upper right")
    ax1.set_ylabel('Frequency')
    ax1.set_xlabel('NTG prediction')
    ax1.set_title("Histogram")
    
    ax2.hist(test_data["Hef_predicted"], density=True, stacked = True, bins=50, label="Hef prediction")

    kde_xs2 = np.linspace(0, 1, 301)
    kde2 = st.gaussian_kde(test_data["Hef_predicted"])
    ax2.plot(kde_xs2, kde2.pdf(kde_xs2), label="Density function")
    ax2.legend(loc="upper right")
    ax2.set_ylabel('Frequency')
    ax2.set_xlabel('Hef prediction')
    ax2.set_title("Histogram")
    f.savefig(ntg_hist_path + "_ntg_hef_histograms.png")

#following three functions are used to convert created predictions to a .txt map, add .irap file header to the map, and then save it in a .irap format
#this is far from optimal, but .irap format is very confusing, so I'm not sure how to improve it
#hard to describe following functions, since they are wrote on a try/fail principle
def convert_prediction_to_map(conf):
    
    import numpy as np
    import pandas as pd
    import os
    
    test_predictions_path = conf['test_predictions_path']
    indices_path = conf["indices_path"]
    prediction_map_path_mean = conf["prediction_map_path_mean"]
    prediction_map_path_p10 = conf["prediction_map_path_p10"]
    prediction_map_path_p50 = conf["prediction_map_path_p50"]
    prediction_map_path_p90 = conf["prediction_map_path_p90"]    
    
    indices = np.load(indices_path)
    predictions = pd.read_csv(test_predictions_path, index_col = 0)
    
    na_value = 9999900.000000
    hef_pred = predictions["Hef_predicted"]
    hef_pred_p10 = predictions["Hef_predicted_p10"]
    hef_pred_p50 = predictions["Hef_predicted_p50"]
    hef_pred_p90 = predictions["Hef_predicted_p90"]
    
    hef_map = np.repeat(na_value, len(indices))
    hef_map_p10 = np.repeat(na_value, len(indices))
    hef_map_p50 = np.repeat(na_value, len(indices))
    hef_map_p90 = np.repeat(na_value, len(indices))
    
    count = 0
    for index in range(len(indices)):
        if indices[index] == True:
            hef_map[index] = hef_pred[count]
            hef_map_p10[index] = hef_pred_p10[count]
            hef_map_p50[index] = hef_pred_p50[count]
            hef_map_p90[index] = hef_pred_p90[count]
            count += 1
            
   
    np.savetxt(prediction_map_path_mean, np.reshape(hef_map[:-4], (-1, 6)), fmt = '%7.7f')
    np.savetxt(prediction_map_path_p10, np.reshape(hef_map_p10[:-4], (-1, 6)), fmt = '%7.7f')
    np.savetxt(prediction_map_path_p50, np.reshape(hef_map_p50[:-4], (-1, 6)), fmt = '%7.7f')
    np.savetxt(prediction_map_path_p90, np.reshape(hef_map_p90[:-4], (-1, 6)), fmt = '%7.7f')
    

def map_postprocessing(attribute_path):
    with open(attribute_path, "a") as f:
        f.write("9999900.0000000 9999900.0000000 9999900.0000000 9999900.0000000")
        
    with open(attribute_path, "r+") as f:
        content = f.read()
        f.seek(0,0)
        header = "-996 3041 20.000000 20.000000" + "\n" + "422756.578778 484576.578778 5036943.474278 5097743.474278" + "\n" + "3092 0.000000 422756.578778 5036943.474278" + "\n" + "0 0 0 0 0 0 0" + "\n"
        
        f.write(header + content)
        
    
def save_irap_file(prediction_map_path,
                     prediction_irap_path):
    import numpy as np
    
    hef = np.array(irap_cps3_reader(prediction_map_path))
    
    hef_np = np.reshape((np.reshape(hef, (3041, 3092), order = 'F')), (3092, 3041))
    np.savetxt(prediction_irap_path, np.reshape(np.reshape(hef_np, -1)[:-4], (-1, 6)), fmt = '%7.7f')
    
    
#function used for plotting interactive maps
#again, due to confusion surrounding .irap format, very hard to explain in details
#this is inherited function from gazprom, I think, with some tweaks
def partial_irap_plot(map_id, conf, order = 'F', limits = None):
    import numpy as np
    import pandas as pd
    
    #map_id takes one of four values - Htot (for plotting total thickness), p10, p50 or p90 (for plotting p10, p50 and p90 predictions)
    if map_id == "Htot":
        attribute_path = "attributes/Htot_Bd.txt"
    else:
        attribute_path = conf['prediction_irap_path_'+map_id]
    
    #well coordinates are used for plotting wells on top of the map
    well_coordinates = pd.read_csv("well_data_coordinates.csv")
    well_coordinates['well_id'] = well_coordinates['Well identifier'] + " " + well_coordinates['Surface']
    well_df = well_coordinates[['well_id','X','Y']]
    
    #reading attribute that should be plotted (Htot, p10, p50 or p90)
    tf = open(attribute_path, "r") #открываем файл на чтение
    skip,inldim,xinc,yinc,xmin,xmax,ymin,ymax,crldim,skip,skip,skip = np.array((tf.readline()+ tf.readline() + tf.readline()).strip().replace('\n',' ').replace('  ',' ').split(' ')).astype(np.float)
    tf.readline()#    text_file.write('0 0 0 0 0 0 0\n')
    attribute_map = np.array(tf.read().strip().replace('\n',' ').replace('   ',' ').replace('  ',' ').split(' ')).astype(np.float).reshape((int(crldim),int(inldim)),order=order)
    
    #limits is a tuple of x_min, x_max, y_min, y_max coordinates, for displaying a subplot
    if limits == None:
        found = False
        #test set has lots of missing data, so those points are all skipped by locating top-left and bottom-right angle of area with non-missing data
        for i in range(attribute_map.shape[0]):
            for j in range(attribute_map.shape[1]):
                if attribute_map[i,j] != 9999900.0:
                    left_angle = (i,j)
                    found = True
                if found == True:
                    break
            if found == True:
                break
        found = False
        for i in range(attribute_map.shape[0]-1,0,-1):
            for j in range(attribute_map.shape[1]-1,0,-1):
                if attribute_map[i,j] != 9999900.0:
                    right_angle = (i,j)
                    found = True
                if found == True:
                    break
            if found == True:
                break
    else:
        left_angle = int((limits[0] - xmin) // xinc), int((limits[2] - ymin) // yinc)
        right_angle = int((limits[1] - xmin) // xinc), int((limits[3] - ymin) // yinc)
    
    import plotly.graph_objects as go
    import plotly.express as px
    
    if limits is not None:
        well_df_reduced = well_df[well_df.X.between(limits[0], limits[1])]
        well_df_reduced2 = well_df_reduced[well_df.Y.between(limits[2], limits[3])]
        x_coord = well_df_reduced2['X']
        y_coord = well_df_reduced2['Y']
        names = well_df_reduced2['well_id']
    else:
        x_coord = well_df['X']
        y_coord = well_df['Y']
        names = well_df['well_id']
   
    attribute_map2 = np.flipud(attribute_map[left_angle[0]:right_angle[0], left_angle[1]:right_angle[1]].T)
    attribute_map2[attribute_map2 == 9999900.0] = 0
    
    if limits is not None:
            x_dim, y_dim = attribute_map2.shape
            hef_np = np.reshape((np.reshape(attribute_map2, (y_dim, x_dim), order = 'F')), (x_dim, y_dim))
            path = conf[f"prediction_irap_path_{map_id}"][:-4] + "_partial.txt"
            flat = np.reshape(hef_np, -1)
            np.savetxt(path, np.reshape(flat[:-5], (-1, 6)), fmt = '%7.7f')
            with open(path, "a") as f:
                last_row = f"{flat[-5]} {flat[-4]} {flat[-3]} {flat[-2]} {flat[-1]}"
                f.write(last_row)
        
            with open(path, "r+") as f:
                content = f.read()
                f.seek(0,0)
                header = f"-996 {y_dim} 20.000000 20.000000" + "\n" + f"{limits[0]} {limits[1]} {limits[2]} {limits[3]}" + "\n" + f"{x_dim} 0.000000 {limits[0]} {limits[2]}" + "\n" + "0 0 0 0 0 0 0" + "\n"
                f.write(header + content)

    if map_id == "Htot":
        fig = px.imshow(attribute_map2, 
                    labels = dict(x = None, y = None, color = map_id), 
                    color_continuous_scale=px.colors.sequential.YlOrBr)
    else:
        fig = px.imshow(attribute_map2, 
                    labels = dict(x = None, y = None, color = map_id), 
                    color_continuous_scale=px.colors.sequential.YlOrBr,
                    range_color = [0,100])
        
    
    fig.add_trace(go.Scatter(x = (x_coord - xmin) / xinc - left_angle[0], 
                             y = right_angle[1] - (y_coord - ymin) / yinc, 
                             hovertemplate = '<b>%{text}</b>',
                             text = names,
                             marker=dict(color='black', size=7), mode = "markers"))
    fig.update_layout(title_text = map_id, title_font_size = 30)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.show()
    

#this is overridden by previous function
def irap_plot(map_id, conf, order = 'F'):
    
    import numpy as np
    import pandas as pd
    
    if map_id == "Htot":
        attribute_path = "attributes/Htot_Bd.txt"
    else:
        attribute_path = conf['prediction_irap_path_'+map_id]
    
    well_coordinates = pd.read_csv("well_data_coordinates.csv")
    well_coordinates['well_id'] = well_coordinates['Well identifier'] + " " + well_coordinates['Surface']
    well_df = well_coordinates[['well_id','X','Y']]
    
    tf = open(attribute_path, "r") #открываем файл на чтение
    skip,inldim,xinc,yinc,xmin,xmax,ymin,ymax,crldim,skip,skip,skip = np.array((tf.readline()+ tf.readline() + tf.readline()).strip().replace('\n',' ').replace('  ',' ').split(' ')).astype(np.float)
    tf.readline()#    text_file.write('0 0 0 0 0 0 0\n')
    attribute_map = np.array(tf.read().strip().replace('\n',' ').replace('   ',' ').replace('  ',' ').split(' ')).astype(np.float).reshape((int(crldim),int(inldim)),order=order)
    
    
    #attribute_map[attribute_map == 9999900.0] = 0
    
    #found = False
    #for i in range(attribute_map.shape[0]):
    #    for j in range(attribute_map.shape[1]):
    #        if attribute_map[i,j] != 9999900.0:
    #            left_angle = (i,j)
    #            found = True
    #        if found == True:
    #            break
    #    if found == True:
    #        break
    #found = False
    #for i in range(attribute_map.shape[0]-1,0,-1):
    #    for j in range(attribute_map.shape[1]-1,0,-1):
    #        if attribute_map[i,j] != 9999900.0:
    #            right_angle = (i,j)
    #            found = True
    #        if found == True:
    #            break
    #    if found == True:
    #        break
    
    import plotly.graph_objects as go
    import plotly.express as px
    
    x_crop_min = 440440
    x_crop_max = 452700
    y_crop_min = 5070300
    y_crop_max = 5077600
    
    left_angle = int((x_crop_min - xmin) // xinc), int((y_crop_min - ymin) // yinc)
    right_angle = int((x_crop_max - xmin) // xinc), int((y_crop_max - ymin) // yinc)
    
    well_df_reduced = well_df[well_df.X.between(x_crop_min, x_crop_max)]
    well_df_reduced2 = well_df[well_df.Y.between(y_crop_min, y_crop_max)]
    x_coord = well_df_reduced2['X']
    y_coord = well_df_reduced2['Y']
    names = well_df_reduced2['well_id']
   
    attribute_map2 = np.flipud(attribute_map[left_angle[0]:right_angle[0], left_angle[1]:right_angle[1]].T)
    attribute_map2[attribute_map2 == 9999900.0] = 0
    if map_id == "Htot":
        fig = px.imshow(attribute_map2, 
                    labels = dict(x = None, y = None, color = map_id), 
                    color_continuous_scale=px.colors.sequential.YlOrBr)
    else:
        fig = px.imshow(attribute_map2, 
                    labels = dict(x = None, y = None, color = map_id), 
                    color_continuous_scale=px.colors.sequential.YlOrBr,
                    range_color = [0,100])
        
    
    fig.add_trace(go.Scatter(x = (x_coord - xmin) / xinc - left_angle[0], 
                             y = right_angle[1] - (y_coord - ymin) / yinc, 
                             hovertemplate = '<b>%{text}</b>',
                             text = names,
                             marker=dict(color='black', size=7), mode = "markers"))
    fig.update_layout(title_text = map_id, title_font_size = 30)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.show()
    
