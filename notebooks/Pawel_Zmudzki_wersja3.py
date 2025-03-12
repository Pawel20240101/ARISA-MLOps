from ARISA_DSML import config, data_preprocessing, model_training, visualization

config.setup_environment()
data_preprocessing.download_and_prepare_data()
X_train, X_test, y_train, y_test = data_preprocessing.prepare_data()
model = model_training.train_model(X_train=config.X_train, y_train=config.y_train)
results = model_training.evaluate_model(model, config.X_test, config.y_test)
print(results)
visualization.plot_results(model, config.X_test, config.y_test)

from ARISA_DSML import config, data_preprocessing, kaggle_setup
config.setup_environment()
kaggle_setup.authenticate_kaggle()
data_preprocessing.download_titanic_data()
X_train, X_test, y_train, y_test = data_preprocessing.prepare_data()
model = model_training.train_model(X_train, y_train)
model_training.evaluate_model(model, X_test, y_test)
