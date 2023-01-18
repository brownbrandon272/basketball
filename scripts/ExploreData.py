import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split
import typing as t
import traceback
from feature_engine.encoding import  RareLabelEncoder
from feature_engine.imputation import CategoricalImputer

import statsmodels.api as sm
from statsmodels.formula.api import ols

from scipy.stats import chi2_contingency

def is_none_or_negative(value) -> bool:
    """
    Checks whether the value is none or negative.

    Parameters
    ----------
    value: `int`, `float`, `None`
        The value to check.

    Returns
    -------
    `bool`
        Whether the value is none or negative.
    """
    if value is None:
        return True
    elif value <= 0:
        return True
    return False

def encode_rare_labels(cat_vars: t.List[str], data:pd.DataFrame, tol:float=0.02) -> pd.DataFrame:
    df = data.copy()

    for var in cat_vars:
        assert var in df.columns, f"Column {var} does not exist in dataframe"
        df[var] = df[var].astype('object')
        # print(f"{var} data type: {df[var].dtype}")
    
    # cat_vars = ['class', 'who', 'survived'] #, 'pclass', 'sibsp', 'parch']
    # df = df[cat_vars]

    ## replace missing values with new label: "Missing"
    # set up the class
    cat_imputer_missing = CategoricalImputer(
        imputation_method='missing', variables=cat_vars, return_object=True)

    # fit the class to the train set
    cat_imputer_missing.fit(df)

    # # the class learns and stores the parameters
    # print(cat_imputer_missing.imputer_dict_)

    # replace NA by missing
    df = cat_imputer_missing.transform(df)

    ## Removing Rare Labels
    rare_encoder = RareLabelEncoder(tol=tol, n_categories=1, variables=cat_vars)

    # find common labels
    rare_encoder.fit(df)

    # # the common labels are stored, we can save the class
    # print(rare_encoder.encoder_dict_)

    df = rare_encoder.transform(df) # returns dataframe, now with the inputted labels encode

    return df

class SeabornFig2Grid():
    """
    This class is used to convert a seaborn figure to a matplotlib grid
    """
    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
            isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())


class ExploreData:
    def __init__(self, data:pd.DataFrame, target:str=None, cat_limit:int=10, default_tol:float=0.01, var_dict:dict=None) -> None:
        self.data = data
        self.target = target
        self.cat_limit = cat_limit if cat_limit > 0 else 10
        self.default_tol = default_tol if default_tol > 0 else 0.01
        self.var_dict, _ = self.var_dict_checks(var_dict=var_dict, cat_limit=self.cat_limit)

    def var_dict_checks(self, var_dict:dict=None, cat_limit:int=None) -> t.Tuple[dict, bool]:
        if var_dict is None: var_dict = self.create_var_dict(cat_limit=cat_limit)
        
        try:
            assert "num_vars" in var_dict.keys(), "var_dict must contain 'num_vars'"
            assert isinstance(var_dict["num_vars"], list), "'num_vars' in var_dict must be a list of strings"
            assert "cat_vars" in var_dict.keys(), "var_dict must contain 'cat_vars'"
            assert isinstance(var_dict["cat_vars"], list), "'cat_vars' in var_dict must be a list of strings"
            return var_dict, True
        except:
            var_dict = self.create_var_dict(cat_limit=cat_limit)
            return var_dict, False

    def create_var_dict(self, cat_limit:int=None):
        if is_none_or_negative(cat_limit): cat_limit = self.cat_limit
        var_dict = {"num_vars": list(), "cat_vars": list()}
        for c in self.data.columns:
            col = pd.Series(self.data[c])
            if (col.dtype == 'O') | (len(col.unique()) <= cat_limit):
                var_dict["cat_vars"].append(c)
            else:
                var_dict["num_vars"].append(c)
        return var_dict

    def help(self):
        print("""
        ========= ExploreData Methods =========
        summary(display_limit:int = 7)
            - Prints a summary of the data. Good for looking over the fields in your data.
            - Inputs:
                > display_limit: maximum number of unique records shown for categorical variables.

        target_variable(target:str, var_dict:dict=None, cat_limit:int=None)
            - Full breakdown of the target variable.
                Missing value, univariate and multivariate plots of the target variable.
            - Inputs:
                > target: name of the target variable.
                > var_dict: pass to explicity state the categorical and numerical variables.
                    format - {"cat_vars": list(), "num_vars": list()}
                > cat_limit: Used to create var_dict if none is passed.
                    If a numeric variable has less unique values than this value then it is decided to be a categorical variable.
                    Default is self.cat_limit, which is defined in __init__ as having a default of 10.
                
        missing_values(target:str=None)
            - Shows which fields in data have missing values.
            - Inputs:
                > target: name of the target variable.
                    If target is passed then mean target value

        univariate_charts(var_dict:dict=None, cat_limit:int=None, tol:float=None, target:str=None)
            - Shows univariate plots for all variables in data.
            - Inputs:
                > var_dict: pass to explicity state the categorical and numerical variables.
                    format - {"cat_vars": list(), "num_vars": list()}
                > cat_limit: Used to create var_dict if none is passed.
                    If a numeric variable has less unique values than this value then it is decided to be a categorical variable.
                    Default is self.cat_limit, which is defined in __init__ as having a default of 10.
                > tol: the minimum frequency a label should have to be considered frequent.
                    Categories with frequencies lower than tol will be grouped. Default=0.02
                > target: name of the target variable.
                    If target is passed then only target variable plots are shown.
            
        multivariate_charts(var_dict:dict=None, tol:float=None, target:str=None,
                corr_threshold:float=0.4, p_val_threshold:float=0.01, top_n:int=10)
            - Shows best multivariate plots for all variables in data.  If target is not
                passed, then a relationship metric is used for each plot type, then the 'top_n' 
                plots are shown that meet the relationship metric.
            - How 'best' plots are decided:
                > For numeric pairs, best is decided by correlation.
                > For numeric vs categorical pairs, best is decided by the p-value associated with a One Way ANOVA test.
                > For categorical pairs, best is decided by the p-value associated with a Chi-Squared test.
            - Inputs:
                > var_dict: pass to explicity state the categorical and numerical variables.
                    format - {"cat_vars": list(), "num_vars": list()}
                > tol: the minimum frequency a label should have to be considered frequent.
                    Categories with frequencies lower than tol will be grouped. Default=0.02
                > target: name of the target variable.
                    If target is passed then only target variable plots are shown.
                > corr_threshold: default=0.4
                > p_val_threshold: default=0.01
                > top_n: default=10
        """)

    def summary(self, display_limit:int = 7):
        print("========= Summary Statistics =========")
        print(f"{self.data.shape[0]} rows, {self.data.shape[1]} columns")
        print()

        print("========= Column Level Statistics =========")
        for c in self.data.columns:
            col = pd.Series(self.data[c])
            print(c, '--', col.dtype)
            if c in self.var_dict["cat_vars"]:
                try:
                    print(col.value_counts(dropna = False)[:display_limit])
                except:
                    print(col.value_counts(dropna = False))
            elif c in self.var_dict["num_vars"]:
                print(col.describe(percentiles = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99], datetime_is_numeric=True))
                _ = self.find_outliers_IQR(col)
            else:
                print("Column not in cat_vars or num_vars")
            print('Unique values:', len(col.unique()))
            print('Null count:', sum(col.isna()))
            print()

        print("To see more functions, run help() method")
        
        ## to add?:
        ##  - charts
        ##  - show extra rows if object field is a categorical field (i.e. not a primary or foreign key)
        return

    def find_outliers_IQR(self, series:pd.Series):
        q1=series.quantile(0.25)
        q3=series.quantile(0.75)
        IQR=q3-q1
        outliers = series[((series<(q1-1.5*IQR)) | (series>(q3+1.5*IQR)))]
        print("number of outliers: "+ str(len(outliers)))
        if outliers.any():
            print("max outlier value: "+ str(outliers.max()))
            print("min outlier value: "+ str(outliers.min()))
        return outliers

    def missing_values(self, target:str=None):
        def plot_na(var, target):
            df = self.data.copy()
            var_missing_name = f'{var} missing'
            df[var_missing_name] = df[var].isna()
            
            if target in self.var_dict['cat_vars']:
                ## categorical -- show class distribution by whether value is missing
                df1 = df.groupby(var_missing_name)[target].value_counts(normalize=True)
                df1 = df1.mul(100)
                df1 = df1.rename('percent').reset_index()

                g = sns.catplot(x=var_missing_name,y='percent',hue=target,kind='bar',data=df1)
                g.ax.set_ylim(0,100)

                for p in g.ax.patches:
                    txt = str(p.get_height().round(2)) + '%'
                    txt_x = p.get_x() 
                    txt_y = p.get_height()
                    g.ax.text(txt_x,txt_y,txt)
                
            elif target in self.var_dict['num_vars']:
                ## Numeric -- show mean target value by whether value is missing
                # Faster to do aggregations before calling sns.barplot, but we get no error bar
                staging = df[[var_missing_name, target]].groupby(var_missing_name)[target].agg(['mean', 'std'])
                sns.barplot(y = staging['mean'], x = staging.index)

            missing_count = df[var_missing_name].sum()
            value_count = df.shape[0] - missing_count
            missing_pct = missing_count/(df.shape[0])
            plt.title(f"Mean {target} by {var} status\n{value_count} values vs {missing_count} missing ({missing_pct:.2%})")
            plt.show()
            print(f"Missing values in {var} in data: {missing_count} ({missing_pct:.2%})\n")
            return

        if target is None: target = self.target
        
        vars_w_na = self.data.isna().sum()[self.data.isna().sum() > 0].index
        print(f'{len(vars_w_na)} variables with missing values: {list(vars_w_na)}')

        self.data[vars_w_na].isna().mean().sort_values(ascending=False).plot.bar(figsize=(10, 4))
        plt.ylabel('Percentage of missing data')
        plt.axhline(y=0.90, color='r', linestyle='-')
        plt.axhline(y=0.80, color='g', linestyle='-')
        plt.show()

        if target is None: return
        for var in vars_w_na:
            plot_na(var=var, target=target)
        return

    def target_variable(self, target:str, var_dict:dict=None, cat_limit:int=None):
        assert target in self.data.columns, f"target variable {target} not in data"
        if is_none_or_negative(cat_limit): cat_limit = self.cat_limit
        
        try:
            var_dict, is_valid_var_dict = self.var_dict_checks(var_dict=var_dict)
            if not is_valid_var_dict: return

            ## Start with comparing missing values to target variable
            self.missing_values(target=target)
            
            col = pd.Series(self.data[target])
            print(target, '--', col.dtype)

            self.univariate_charts(target=target, var_dict=var_dict, cat_limit=cat_limit)
            self.multivariate_charts(target=target, var_dict=var_dict)
        except Exception as e:
            traceback.print_exc()
            
        return

    def univariate_charts(self, var_dict:dict=None, cat_limit:int=None, tol:float=None, target:str=None):
        def numeric_charts(c:str, df:pd.DataFrame):
            def stratify_split(df:pd.DataFrame):
                ## stratify split on continuous variable
                # Split number of observation into bins
                bins = np.linspace(0, df.shape[0], 30)

                # Save your values in a new ndarray, broken down by the bins created above.
                y_binned = np.digitize(df[c], bins)

                try:
                    # Pass y_binned to the stratify argument, and sklearn will handle the rest
                    _, _, _, y_test = \
                    train_test_split(df,
                                    df[c],
                                    test_size = 250.0 / df.shape[0],
                                    stratify=y_binned)
                except:
                    # Stratify requires a min occurrence of two
                    print(f"Error stratify sampling {c} for violin plot; now doing a random sample")
                    _, _, _, y_test = \
                    train_test_split(df,
                                    df[c],
                                    test_size = 250.0 / df.shape[0])
                return y_test

            fig, ax = plt.subplots(1, 4, figsize = (16, 4))
            plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)

            df[c].hist(ax = ax[0])
            ax[0].set_xlabel(c)
            ax[0].set_title('histogram')

            sns.distplot(x = df[c], ax = ax[1])
            ax[1].set_xlabel(c)
            ax[1].set_title('histogram + density')

            if df.shape[0] <= 250:
                y_test = df[c]
            else:
                y_test = stratify_split(df)

            sns.violinplot(x = df[c], ax = ax[2], palette = 'vlag')
            sns.swarmplot(y_test, ax = ax[2])
            ax[2].set_xlabel(c)
            ax[2].set_title('violin + swarm plot')

            sns.boxplot(x = df[c], ax = ax[3])
            ax[3].set_xlabel(c + ' - boxplot')
            ax[3].set_title('boxplot')

            plt.show()
        def categorical_charts(c:str, df:pd.DataFrame):
            counts = df[c].value_counts(dropna = False)
            
            fig, ax = plt.subplots(1, 2, figsize = (10, 4))
            plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)

            _labels = df[c].value_counts().index
            sns.countplot(x = df[c], order = _labels, ax = ax[0])
            if len(df[c].unique()) >= 5:
                ax[0].set_xticklabels(labels = _labels, rotation = 40, ha = 'right')
            else:
                ax[0].set_xticklabels(labels = _labels)

            center_circle = plt.Circle((0, 0), 0.70, fc='white')
            ax[1].pie(counts.values, labels = counts.index.astype(str))
            ax[1].add_artist(center_circle)
            ax[1].set_title(c)
            # ax[1].set_xticklabels(labels = counts.index.astype(str), rotation = 40)

            plt.show()
            return

        if is_none_or_negative(tol): tol=self.default_tol
        if is_none_or_negative(cat_limit): cat_limit=self.cat_limit

        var_dict, is_valid_var_dict = self.var_dict_checks(var_dict=var_dict, cat_limit=cat_limit)
        if not is_valid_var_dict: return

        if var_dict["cat_vars"]:
            df = encode_rare_labels(cat_vars=var_dict["cat_vars"], data=self.data, tol=tol)
        else:
            df = self.data.copy()

        for c in var_dict["num_vars"]:
            if target is not None:
                if c != target: continue
            numeric_charts(c, df)
            
        for c in var_dict["cat_vars"]:
            if target is not None:
                if c != target: continue
            categorical_charts(c, df)

        return

    def multivariate_charts(self, 
            var_dict:dict=None, tol:float=None, target:str=None,
            corr_threshold:float=0.4, p_val_threshold:float=0.01, top_n:int=10
            ) -> None:
        def numeric_pairs(var_dict, df:pd.DataFrame, corr_threshold, top_n, target:str=None):
            def get_high_corr_pairs(var_dict, df, corr_threshold, top_n, target:str=None):
                high_corr_pairs = set()

                # df = df.copy()
                # if target is not None:
                #     # Put target column in the beginning
                #     target_column = df.pop(target)
                #     df.insert(0, target, target_column)

                corr_matrix = df[var_dict["num_vars"]].corr()
                corr_matrix = corr_matrix.mask(np.tril(np.ones(corr_matrix.shape)).astype(np.bool))
                    ## ===== Use this to turn bottom triangle of correlation matrix to NaNs =====
                    # print (np.tril(np.ones(corr_matrix.shape)))
                    #  [[ 1.  0.  0.  0.  0.  0.]
                    #   [ 1.  1.  0.  0.  0.  0.]
                    #   [ 1.  1.  1.  0.  0.  0.]
                    #   [ 1.  1.  1.  1.  0.  0.]
                    #   [ 1.  1.  1.  1.  1.  0.]
                    #   [ 1.  1.  1.  1.  1.  1.]]

                mtx_unstack = corr_matrix.unstack().dropna()
                mtx_unstack_sorted = mtx_unstack.abs().sort_values(ascending=False)
                
                ## Get top correlation pairs
                if target is None:
                    # If there is no target, use top correlation pairs
                    min_corr_threshold = min(corr_threshold, mtx_unstack_sorted.values.max())
                    top10 = mtx_unstack_sorted[mtx_unstack_sorted >= min_corr_threshold].head(top_n)
                    high_corr_pairs = pd.concat([mtx_unstack, top10], axis=1, join="inner").reset_index()
                    high_corr_pairs.columns = ["x", "y", "correlation", "abs_correlation"]
                else:
                    # If there is a target, use target column with all other features
                    high_corr_pairs = pd.concat([mtx_unstack, mtx_unstack_sorted], axis=1, join="inner").reset_index()
                    high_corr_pairs.columns = ["x", "y", "correlation", "abs_correlation"]
                    high_corr_pairs = high_corr_pairs[(high_corr_pairs['y'] == target) | (high_corr_pairs['x'] == target)]

                ## Create value counts of bins
                bins = np.arange(0.0, 1.1, 0.1)
                binned_correlations = pd.cut(mtx_unstack_sorted, bins=bins).value_counts(normalize=True).sort_index(ascending=False)
                
                return high_corr_pairs, binned_correlations, corr_matrix
            

            high_corr_pairs, binned_correlations, corr_matrix = \
                get_high_corr_pairs(var_dict=var_dict, df=df, corr_threshold=corr_threshold, top_n=top_n, target=target)
            print("\nhigh_corr_pairs\n", high_corr_pairs)
            print("\nbinned_correlations\n", binned_correlations)
            
            df_ = df.copy()
            ## Visualisations can be slow with too many ovservations
            if df_.shape[0] > 10000: df_ = df_.sample(n=10000)
            
            for i, (x, y, corr_val, _) in high_corr_pairs.iterrows():
                print(f"num vars:{x}, {y}")
                ## having trouble plotting them side-by-side, just do them on top of eachother
                g0 = sns.jointplot(x=x, y=y, data=df_, kind='scatter')
                g1 = sns.jointplot(x=x, y=y, data=df_, kind='kde', fill=True)
                if corr_val < 0:
                    g1.ax_joint.annotate(f'corr = {corr_val:.3f}', xy=(0.98, 0.9), xycoords='axes fraction',
                                        ha='right', va='center')
                else:
                    g1.ax_joint.annotate(f'corr = {corr_val:.3f}', xy=(0.02, 0.9), xycoords='axes fraction',
                                        ha='left', va='center')
                fig = plt.figure(figsize=(11,6))
                gs = gridspec.GridSpec(1, 2)

                mg0 = SeabornFig2Grid(g0, fig, gs[0])
                mg1 = SeabornFig2Grid(g1, fig, gs[1])

                gs.tight_layout(fig)
                plt.show()
            return
        def boxplots(var_dict, df:pd.DataFrame, p_val_threshold, top_n, target:str=None):
            def get_anova_pairs(var_dict, p_val_threshold, top_n, target:str=None) -> pd.DataFrame:
                anova_pairs = list()
                cat_vars = [target] if target in var_dict['cat_vars'] else var_dict['cat_vars']
                num_vars = [target] if target in var_dict['num_vars'] else var_dict['num_vars']
                
                for cat_var in cat_vars:
                    for num_var in num_vars:
                        try:
                            mod = ols(f"{num_var} ~ {cat_var}", data=df).fit()
                        except Exception as e:
                            print(f"Error occurred fitting Ordinary Least Squares to {cat_var} and {num_var}: {e}")
                            continue
                        try:
                            aov_table = sm.stats.anova_lm(mod, typ=2)
                        except Exception as e:
                            print(f"Error running ANOVA LM on {cat_var} and {num_var}: {e}")
                            continue
                        anova_pairs.append((cat_var, num_var, aov_table['PR(>F)'][0])) # the p value

                anova_pairs = pd.DataFrame(anova_pairs, columns=['cat_var', 'num_var', 'anova_p_value'])
                anova_pairs.sort_values('anova_p_value', ascending=True, inplace=True)
                if target is not None:
                    anova_pairs = anova_pairs[(anova_pairs['num_var'] == target) | (anova_pairs['cat_var'] == target)]
                else:
                    anova_pairs = anova_pairs[anova_pairs['anova_p_value'] <= p_val_threshold].head(top_n)

                return anova_pairs
            
            anova_pairs = get_anova_pairs(var_dict=var_dict, p_val_threshold=p_val_threshold, top_n=top_n, target=target)
            
            for i, (cat_var, num_var, anova_p_val) in anova_pairs.iterrows():
                sns.boxplot(data=df, x=cat_var, y=num_var).set(title=f"{cat_var} vs {num_var}: ANOVA p-val {anova_p_val:.2E}")
                plt.show()
            return
        def co_ocurrence(var_dict, df:pd.DataFrame, p_val_threshold, top_n, target:str=None):
            def get_chi2_pairs(var_dict, p_val_threshold, top_n, target:str=None) -> pd.DataFrame:
                chi2_pairs = list()
                cat_var_count = len(var_dict['cat_vars'])
                for x in range(cat_var_count):
                    for y in range(x+1, cat_var_count):
                        if target is not None and target not in [x, y]: continue

                        cat_var_x = var_dict['cat_vars'][x]
                        cat_var_y = var_dict['cat_vars'][y]

                        contingency = pd.crosstab(df[cat_var_x], df[cat_var_y])
                        chi2, p_val, deg_of_free, expected_freq = chi2_contingency(contingency)
                        chi2_pairs.append((cat_var_x, cat_var_y, p_val))

                chi2_pairs = pd.DataFrame(chi2_pairs, columns=['cat_var_x', 'cat_var_y', 'chi2_p_value'])
                chi2_pairs.sort_values('chi2_p_value', ascending=True, inplace=True)
                if target is not None:
                    chi2_pairs = chi2_pairs[(chi2_pairs['cat_var_x'] == target) | (chi2_pairs['cat_var_y'] == target)]
                else:
                    chi2_pairs = chi2_pairs[chi2_pairs['chi2_p_value'] <= p_val_threshold].head(top_n)

                return chi2_pairs

            chi2_pairs = get_chi2_pairs(var_dict=var_dict, p_val_threshold=p_val_threshold, top_n=top_n, target=target)
            
            for i, (cat_var_x, cat_var_y, chi2_p_val) in chi2_pairs.iterrows():
                contingency = pd.crosstab(df[cat_var_x], df[cat_var_y])
                contingency_pct = pd.crosstab(df[cat_var_x], df[cat_var_y], normalize='index')
                annot=contingency.astype(str) + "\n" + contingency_pct.multiply(100).round(1).astype(str) + "%"
                sns.heatmap(contingency_pct, annot=annot, fmt='', cmap="YlGnBu") \
                    .set(title=f"{cat_var_x} vs {cat_var_y}: Chi-Sq p-val {chi2_p_val:.2E}")
                plt.show()
            return
        if is_none_or_negative(tol): tol=self.default_tol
        
        var_dict, is_valid_var_dict = self.var_dict_checks(var_dict=var_dict)
        if not is_valid_var_dict: return

        df = encode_rare_labels(cat_vars=var_dict["cat_vars"], data=self.data, tol=tol)

        ## Categorical vs Numeric plots
        boxplots(var_dict=var_dict, df=df, p_val_threshold=p_val_threshold, top_n=top_n, target=target)

        ## Numeric vs Numeric plots
        if (target is None) or (target in var_dict["num_vars"]):
            numeric_pairs(var_dict=var_dict, df=df, corr_threshold=corr_threshold, top_n=top_n, target=target)
        
        ## Categorical vs Categorical plots
        if (target is None) or (target in var_dict["cat_vars"]):
            co_ocurrence(var_dict=var_dict, df=df, p_val_threshold=p_val_threshold, top_n=top_n, target=target)
        
        return


if __name__ == "__main__":
    df = sns.load_dataset("penguins")
    # df = sns.load_dataset("titanic")
    explore = ExploreData(df)
    # explore.summary()
    explore.multivariate_charts()
    # explore.target_variable('species') # penguins dataset
    explore.target_variable('bill_length_mm') # penguins dataset
    # explore.target_variable('survived') # titanic dataset
