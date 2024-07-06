import pandas as pd
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt

from typing import Union, List

class EasyOLS:
    """
    Easily generate readable conclusions and plot based on OLS summary data.
    Works with 1 dependent variable and 1+ independent variables.
    
    Parameters:
    - dependent_var (str) - Column name of pandas.DataFrame
    - independent_vars (str or list(str)) - Column name(s) of pandas.DataFrame
    - df - (pandas.DataFrame)
    
    Methods:
    - print_summary() - Prints standard statsmodels ols summary and prints human-friendly 
    conclusions below.
    - plot() - Prints scatterplot of original data and predicted values
        - Parameters (optional):
            - title (str)
            - xlabel (str)
            - ylabel (str)
            - description (str)
    
    
    Properties:
    - coefficients (pandas.Series) - Calculated coefficients of intercept and 
    independent variables. [0] returns coeff. of intercept, [n] returns coeff.
    of nth independent variable given when initializing model
    - confidences (pandas.Series) - Calculated confidences of intercept and 
    independent variables. (Ex: 0.6 means 60% confident.) [0] returns conf. of 
    intercept, [n] returns conf. of nth independent variable given when initializing model        
    - dependent_var (str)
    - independent_vars (str or list(str))
    - df (pandas.DataFrame)
    - formula
    - model
    - internal_dependent_var
    - internal_independent_vars

    

    """
    def __init__(self, dependent_var: str, independent_vars: Union[str, List[str]], df):
        if not isinstance(dependent_var, str):
            raise ValueError("dependent_var must be a string")

        if not isinstance(independent_vars, str) and not all(isinstance(var, str) for var in independent_vars):
            raise ValueError("independent_vars must be a string or an array of strings")

        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a DataFrame")

        self.dependent_var = dependent_var
        self.independent_vars = independent_vars
        self.df = df

        self.formula = None
        self.model = None

        # Variables in formula passed to statsmodels.formula.api.ols
        # Contain Q('')
        self.internal_dependent_var = None
        self.internal_independent_vars = None

        self.coefficients = None
        self.confidences = None

        # The order of these functions matters
        self.__create_formula()
        self.__define_model()
        self.__extract_var_names()
        self.__fit()

    def __create_formula(self):
        """
        Creates formula parameter for statsmodels.formula.api.ols
        """
        # https://www.statsmodels.org/stable/generated/statsmodels.formula.api.ols.html
        # Final formula will look like this:
        #   'Q("Foo Bar") ~ Q("Bizz Buzz")'
        #      ^ must be ", CANNOT be '
        # Or, if there are multiple independent variables, final formula will
        # look like this:
        #   'Q("Foo Bar") ~ Q("Bizz Buzz") + Q("Baz - Qux")'
        #
        # statsmodels.formula.api.ols("Foo Bar ~ Bizz Buzz", pandasDataFrame)
        # will raise an error due to spaces in column names.
        #
        # The Q function allows spaces in column names, as long as single
        # quotation marks (') are used.
        # Q('Foo Bar') - WRONG
        # Q{"Foo Bar"} - RIGHT

        # Q("Foo Bar") ~ Q("Bizz Buzz")
        # ^^^^^^^^^^^^
        dependent_part = f"Q(\"{self.dependent_var}\")"

        if isinstance(self.independent_vars, str):
            # A single dependent variable was passed as argument
            # Q("Foo Bar") + Q("Bizz Buzz")'
            #                ^^^^^^^^^^^^^^
            independent_part = f"Q(\"{self.independent_vars}\")"
        else:
            # A list of dependent variables was passed as argument
            # 'Q("Foo Bar") ~ Q("Bizz Buzz") + Q("Baz - Qux")'
            #                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            independent_part = ' + '.join([f"Q(\"{var}\")" for var in self.independent_vars])

        formula = f"{dependent_part} ~ {independent_part}"
        self.formula = formula


    def __define_model(self):
        """
        Creates the OLS model with the given formula and dataframe.
        Does not fit the model.

        """
        self.model = ols(self.formula, self.df)

    def __extract_var_names(self):
        """
        Extracts names of the dependent variable and independent variables from
        the formula. These names must be extracted from the model before it is
        fitted.

        """
        self.internal_dependent_var = self.model.endog_names
        self.internal_independent_vars = self.model.exog_names

    def __fit(self):
        """
        Fits the OLS model. Model must be defined first.

        """
        self.model = self.model.fit()
        self.coefficients = self.model.params
        self.confidences = 1 - self.model.pvalues

    def __format_var(self, var):
        """
        'Q("Foo Bar")' -> 'Foo Bar'

        """
        if var == "Intercept":
            return var

        return var[3:len(var)-2]

    def summary(self):
        # standard statsmodels.formula.api.ols summary
        print(self.model.summary())

        # Append EasyOLS conclusions below
        print("\nConclusions:")

        dependent_var = self.__format_var(self.internal_dependent_var)
        independent_vars = [self.__format_var(var) for var in self.internal_independent_vars]

        areMultipleIndependentVars = False
        # Not counting Intercept as independent variable
        if (len(independent_vars)>2):
            areMultipleIndependentVars = True

        if (areMultipleIndependentVars):
            print(f"Independent variables: {', '.join(independent_vars[1:])}")

        for i in range(0, len(independent_vars)):
            # format as %, round to 2 decimal places
            confidence = '{:.2%}'.format(self.confidences.iloc[i])
            # round to 2 decimal places
            coefficient = "{:.2f}".format(self.coefficients.iloc[i])


            # Definitions from Example 2 from https://www.statology.org/intercept-in-regression/
            if i==0:
                # Intercept: The mean value of the response variable when all
                # predictor variables are zero
                if(areMultipleIndependentVars):
                    print(f"This model is {confidence} confident when all independent variables are 0, the average value of {dependent_var} is {coefficient}.")
                else:
                    print(f"This model is {confidence} confident when {independent_vars[i+1]} is 0, the average value of {dependent_var} is {coefficient}.") # i will only ever be 1 if there are not multiple independent variables

            else:
                # Regular independent variable: The average change in the
                # response variable for a one unit increase in the jth
                # predictor variable, assuming all other predictor variables are held constant
                if(areMultipleIndependentVars):
                    print(f"This model is {confidence} confident increasing {independent_vars[i]} by 1 will, on average, change {dependent_var} by {'+' if float(coefficient) > 0 else ''}{coefficient} when all other independent variables are held constant.")
                else:
                    print(f"This model is {confidence} confident increasing {independent_vars[i]} by 1 will, on average, change {dependent_var} by {'+' if float(coefficient) > 0 else ''}{coefficient}.")
                    
    def plot(self, 
             title = None,
             xlabel = None, 
             ylabel = None,
             description = None):
        if not isinstance(self.independent_vars, str) and hasattr(self.independent_vars, '__iter__') and len(self.independent_vars) > 1:
            raise ValueError("Cannot create plot for models with multiple independent variables.")
        
        df_pred = self.df
        df_pred[f"Predicted {self.dependent_var}"] = self.model.predict(df_pred[self.independent_vars])
        
        full_title = title if title else f"{self.independent_vars} vs. {self.dependent_var}"
        if description:
            full_title += f"\n{description}"
        
        plt.figure()
        plt.title(full_title)
        plt.xlabel(xlabel if xlabel else f"{self.independent_vars}")
        plt.ylabel(ylabel if ylabel else f"{self.dependent_var}")
        
        plt.scatter(df_pred[self.independent_vars], df_pred[self.dependent_var], s=1, color='blue', alpha=0.5)
        plt.scatter(df_pred[self.independent_vars], df_pred[f"Predicted {self.dependent_var}"], s=1, color='red', alpha=0.5)

        plt.axhline(y=0, color='black',linewidth=0.5)
        
        plt.show()
        
