import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
N_SAMPLES = 700
PERCENT_TRAIN = 0.8


class Plotter:

    def __init__(self, regularization, poly_degree, print_images=False):
        self.reg = regularization
        self.POLY_DEGREE = poly_degree
        self.print_images = print_images
        self.rng = np.random.RandomState(seed=10)

    def print_figure(self, figure, title):
        fig_title = title.replace(' ', '_')
        path = f'outputs/{fig_title}.png'
        figure.write_image(path)
        img = mpimg.imread(path)
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    def split_data(self, x_all, y_all):
        rng = self.rng
        all_indices = rng.permutation(N_SAMPLES)
        train_indices = all_indices[:round(N_SAMPLES * PERCENT_TRAIN)]
        test_indices = all_indices[round(N_SAMPLES * PERCENT_TRAIN):]
        xtrain = x_all[train_indices]
        ytrain = y_all[train_indices]
        xtest = x_all[test_indices]
        ytest = y_all[test_indices]
        return xtrain, ytrain, xtest, ytest, train_indices, test_indices

    # def plot_all_data(self, x_all, y_all, p):                                   # change to remove red line?
    #     df = pd.DataFrame({'feature1': x_all[:, 0], 'feature2': x_all[:, 1],
    #         'y': np.squeeze(y_all), 'best_fit': np.squeeze(p)})
    #     title = 'All Simulated Datapoints'
    #     fig = plt.figure(figsize=(10, 8))
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.set_title(title)
    #     ax.scatter(df['feature1'], df['feature2'], df['y'], color='blue', s
    #         =8, alpha=0.12, label='Data Points')
    #     ax.plot(df['feature1'], df['feature2'], df['best_fit'], color='red',
    #         linewidth=2, label='Line of Best Fit')
    #     ax.set_xlim([df['feature1'].min(), df['feature1'].max()])
    #     ax.set_ylim([df['feature2'].min(), df['feature2'].max()])
    #     ax.set_zlim([min(df['y'].min(), df['best_fit'].min()), max(df['y'].
    #         max(), df['best_fit'].max())])
    #     ax.set_xlabel('Feature 1')
    #     ax.set_ylabel('Feature 2')
    #     ax.set_zlabel('Y')
    #     ax.legend()
    #     plt.show()
    #     if self.print_images:
    #         self.print_figure(title)

    def plot_split_data(self, xtrain, xtest, ytrain, ytest):
        ytrain = ytrain.reshape(-1)
        ytest = ytest.reshape(-1)
        train_df = pd.DataFrame({'feature1': xtrain[:, 0], 'feature2':
            xtrain[:, 1], 'y': ytrain, 'label': 'Training'})
        test_df = pd.DataFrame({'feature1': xtest[:, 0], 'feature2': xtest[
            :, 1], 'y': ytest, 'label': 'Testing'})
        title = 'Data Set Split'
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(title)
        ax.scatter(train_df['feature1'], train_df['feature2'], train_df['y'
            ], color='yellow', s=2, alpha=0.75, label='Training')
        ax.scatter(test_df['feature1'], test_df['feature2'], test_df['y'],
            color='red', s=2, alpha=0.75, label='Testing')
        ax.set_xlim([min(train_df['feature1'].min(), test_df['feature1'].
            min()), max(train_df['feature1'].max(), test_df['feature1'].max())]
            )
        ax.set_ylim([min(train_df['feature2'].min(), test_df['feature2'].
            min()), max(train_df['feature2'].max(), test_df['feature2'].max())]
            )
        ax.set_zlim([min(train_df['y'].min(), test_df['y'].min()), max(
            train_df['y'].max(), test_df['y'].max())])
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Y')
        ax.legend()
        plt.show()
        if self.print_images:
            self.print_figure(title)

    def plot_linear_closed(self, xtrain, xtest, ytrain, ytest, x_all, y_pred):
        ytrain = ytrain.reshape(-1)
        ytest = ytest.reshape(-1)
        train_df = pd.DataFrame({'feature1': xtrain[:, 0], 'feature2':
            xtrain[:, 1], 'y': ytrain, 'label': 'Training'})
        test_df = pd.DataFrame({'feature1': xtest[:, 0], 'feature2': xtest[
            :, 1], 'y': ytest, 'label': 'Testing'})
        pred_df = pd.DataFrame({'feature1': x_all[:, 0], 'feature2': x_all[
            :, 1], 'Trendline': np.squeeze(y_pred)})
        title = 'Linear (Closed)'
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(title)
        ax.scatter(train_df['feature1'], train_df['feature2'], train_df['y'
            ], color='yellow', s=2, alpha=0.75, label='Training')
        ax.scatter(test_df['feature1'], test_df['feature2'], test_df['y'],
            color='red', s=2, alpha=0.75, label='Testing')
        ax.plot(pred_df['feature1'], pred_df['feature2'], pred_df[
            'Trendline'], color='red', linewidth=2, label='Trendline')
        ax.set_xlim([x_all[:, 0].min(), x_all[:, 0].max()])
        ax.set_ylim([x_all[:, 1].min(), x_all[:, 1].max()])
        ax.set_zlim([min(ytrain.min(), ytest.min(), pred_df['Trendline'].
            min()), max(ytrain.max(), ytest.max(), pred_df['Trendline'].max())]
            )
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Y')
        ax.legend()
        plt.show()
        if self.print_images:
            self.print_figure(title)

    def plot_linear_gd(self, xtrain, xtest, ytrain, ytest, x_all, y_pred):
        ytrain = ytrain.reshape(-1)
        ytest = ytest.reshape(-1)
        train_df = pd.DataFrame({'feature1': xtrain[:, 0], 'feature2':
            xtrain[:, 1], 'y': ytrain, 'label': 'Training'})
        test_df = pd.DataFrame({'feature1': xtest[:, 0], 'feature2': xtest[
            :, 1], 'y': ytest, 'label': 'Testing'})
        pred_df = pd.DataFrame({'feature1': x_all[:, 0], 'feature2': x_all[
            :, 1], 'Trendline': np.squeeze(y_pred)})
        title = 'Linear (GD)'
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(title)
        ax.scatter(train_df['feature1'], train_df['feature2'], train_df['y'
            ], color='yellow', s=2, alpha=0.75, label='Training')
        ax.scatter(test_df['feature1'], test_df['feature2'], test_df['y'],
            color='red', s=2, alpha=0.75, label='Testing')
        ax.plot(pred_df['feature1'], pred_df['feature2'], pred_df[
            'Trendline'], color='red', linewidth=2, label='Trendline')
        ax.set_xlim([x_all[:, 0].min(), x_all[:, 0].max()])
        ax.set_ylim([x_all[:, 1].min(), x_all[:, 1].max()])
        ax.set_zlim([min(ytrain.min(), ytest.min(), pred_df['Trendline'].
            min()), max(ytrain.max(), ytest.max(), pred_df['Trendline'].max())]
            )
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Y')
        ax.legend()
        plt.show()
        if self.print_images:
            self.print_figure(title)

    def plot_linear_gd_tuninglr(self, xtrain, xtest, ytrain, ytest, x_all,
        x_all_feat, learning_rates, weights):
        ytrain = ytrain.reshape(-1)
        ytest = ytest.reshape(-1)
        train_df = pd.DataFrame({'feature1': xtrain[:, 0], 'feature2':
            xtrain[:, 1], 'y': ytrain, 'label': 'Training'})
        test_df = pd.DataFrame({'feature1': xtest[:, 0], 'feature2': xtest[
            :, 1], 'y': ytest, 'label': 'Testing'})
        title = 'Tuning Linear (GD)'
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(title)
        ax.scatter(train_df['feature1'], train_df['feature2'], train_df['y'
            ], color='yellow', s=2, alpha=0.75, label='Training')
        ax.scatter(test_df['feature1'], test_df['feature2'], test_df['y'],
            color='red', s=2, alpha=0.75, label='Testing')
        colors = ['green', 'blue', 'pink']
        for ii in range(len(learning_rates)):
            y_pred = self.reg.predict(x_all_feat, weights[ii])
            y_pred = np.reshape(y_pred, (y_pred.size,))
            pred_df = pd.DataFrame({'feature1': x_all[:, 0], 'feature2':
                x_all[:, 1], 'Trendline': np.squeeze(y_pred)})
            ax.plot(pred_df['feature1'], pred_df['feature2'], pred_df[
                'Trendline'], color=colors[ii], linewidth=2, label=
                'Trendline LR=' + str(learning_rates[ii]))
        ax.set_xlim([x_all[:, 0].min(), x_all[:, 0].max()])
        ax.set_ylim([x_all[:, 1].min(), x_all[:, 1].max()])
        ax.set_zlim([min(ytrain.min(), ytest.min(), pred_df['Trendline'].
            min()), max(ytrain.max(), ytest.max(), pred_df['Trendline'].max())]
            )
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Y')
        ax.legend()
        plt.show()
        if self.print_images:
            self.print_figure(title)

    # def plot_10_samples(self, x_all, y_all_noisy, sub_train, y_pred, title):
    #     samples_df = pd.DataFrame({'feature1': x_all[sub_train, 0],
    #         'feature2': x_all[sub_train, 1], 'y': np.squeeze(y_all_noisy[
    #         sub_train]), 'label': 'Samples'})
    #     pred_df = pd.DataFrame({'feature1': x_all[:, 0], 'feature2': x_all[
    #         :, 1], 'Trendline': np.squeeze(y_pred)})
    #     fig = plt.figure(figsize=(10, 8))
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.set_title(title)
    #     ax.scatter(samples_df['feature1'], samples_df['feature2'],
    #         samples_df['y'], color='red', s=30, alpha=0.75, label='Samples')
    #     mask = (pred_df['feature1'] >= samples_df['feature1'].min()) & (pred_df
    #         ['feature1'] <= samples_df['feature1'].max()) & (pred_df[
    #         'feature2'] >= samples_df['feature2'].min()) & (pred_df[
    #         'feature2'] <= samples_df['feature2'].max())
    #     ax.plot(pred_df['feature1'][mask], pred_df['feature2'][mask],
    #         pred_df['Trendline'][mask], color='blue', linewidth=2, label=
    #         'Trendline')
    #     z_min = max(min(samples_df['y'].min(), pred_df['Trendline'].min()),
    #         -1000)
    #     z_max = max(samples_df['y'].max(), pred_df['Trendline'].max())
    #     ax.set_xlim([samples_df['feature1'].min(), samples_df['feature1'].
    #         max()])
    #     ax.set_ylim([samples_df['feature2'].min(), samples_df['feature2'].
    #         max()])
    #     ax.set_zlim([z_min, z_max])
    #     ax.set_xlabel('Feature 1')
    #     ax.set_ylabel('Feature 2')
    #     ax.set_zlabel('Y')
    #     ax.legend()
    #     plt.show()
    #     if self.print_images:
    #         self.print_figure(title)
