import numpy as np


class CategoryStruct:
    """
    This is an abstract class containing some methods to identify structure among the categorical columns in a pandas dataframe.

    By structure we mean that the unique cases defined by a smaller subset of the categorical columns are the same as the unique
    cases defined by all the categorical columns, which implies that some columns could be removed.

    The simplest examples could be: The week day in German does not add any information when we already have it in English. The
    column set {DayEnglish, DayGerman} has the same information as {DayEnglish} (or as {DayGerman} obviously).
    A variable "province" does not add any information that is not already in "postal_code" (assuming postal codes are finer grained
    divisions that subdivide provinces.)

    The decision of what (if any) columns should be removed is not made by the methods. The methods just provide information to
    assist decision (see the individual docstrings).
    """

    @staticmethod
    def categoricals(data):
        """
        Returns a list of column names whose type is a Python object. Categoricals encoded as strings are Python objects in Pandas.
        In all the class methods, when for the argument `categoricals` is not given, this method will be called to define what the
        categorical variables are.

        Args:
            data (pandas.dataframe): The dataframe

        Returns:
            A list with the names of the columns whose type is a Python object
        """
        return [
            name
            for name, type in zip(data.columns, data.dtypes)
            if type == np.dtype("object")
        ]

    @staticmethod
    def cardinality_set_of_rows(data, col_names):
        """
        Returns the cardinality of the set of rows (for the given columns in col_names).

        This function is the workhorse of the whole CategoryStruct class. Everything is just about comparing the cardinalities of
        different sets of columns.

        Args:
            data (pandas.dataframe): The dataframe
            col_names(list of str):	 A list with the names of the columns to be selected

        Returns:
            The cardinality of the set of rows at the specified columns
        """
        if len(col_names) == 0:
            return 1

        return data.drop_duplicates(subset=col_names).shape[0]

    @staticmethod
    def synonyms(data, categoricals=None):
        """
        Searches for variable pairs that contain the same information.

        Args:
            data (pandas.dataframe):	The dataframe
            categoricals(list of str):	A list with the names of all the categorical columns to be searched

        Returns:
            A list of tuples of variable name pairs that contain the same information, typically encoded differently.
        """
        if categoricals is None:
            categoricals = CategoryStruct.categoricals(data)

        ret = []
        if len(categoricals) < 2:
            return ret

        card_name = []
        for name in categoricals:
            card_name.append(
                (CategoryStruct.cardinality_set_of_rows(data, [name]), name)
            )

        card_name.sort()

        for i, (c, name) in enumerate(card_name):
            for j in range(i + 1, len(card_name)):
                if card_name[j][0] != c:
                    break

                candidate = (name, card_name[j][1])
                if CategoryStruct.cardinality_set_of_rows(data, list(candidate)) == c:
                    ret.append(candidate)

        return ret

    @staticmethod
    def all_complete_sets(data, categoricals=None):
        """
        Returns all the complete sets of variables that have the same information as the given set.

        **Note**: This method is the slowest of all, but it already returns all the possible final choices of
        categorical variables that contain the same information as the given set.

        Args:
            data (pandas.dataframe):	The dataframe
            categoricals(list of str):	A list with the names of all the categorical columns to be searched

        Returns:
            A list of lists of the names of variables that are complete sets. This includes the original set passed to the function.
        """
        if categoricals is None:
            categoricals = CategoryStruct.categoricals(data)

        ret = [categoricals]
        if len(categoricals) < 2:
            return ret

        card_all_in = CategoryStruct.cardinality_set_of_rows(data, categoricals)

        for i, col in enumerate(categoricals):
            excluded = [col]
            remaining = [c for c in categoricals if c not in excluded]
            card_others = CategoryStruct.cardinality_set_of_rows(data, remaining)

            if card_others == card_all_in:
                ret.append(remaining)

                for j in range(i + 1, len(categoricals)):
                    excluded.append(categoricals[j])
                    remaining = [c for c in categoricals if c not in excluded]
                    card_others = CategoryStruct.cardinality_set_of_rows(
                        data, remaining
                    )

                    if card_others == card_all_in:
                        ret.append(remaining)
                    else:
                        excluded.pop()

        return ret

    @staticmethod
    def individually_redundant(data, categoricals=None):
        """
        Returns all the variables that could **individually** be removed.

        **Note that**: any of the variables returned can be removed individually. Meaning that only one of them should be removed
        at each step and this method should be called again on the resulting set of columns before removing another variable.
        Removing any variable may change the possible redundancy of the remaining variables.

        Args:
            data (pandas.dataframe):	The dataframe
            categoricals(list of str):	A list with the names of all the categorical columns to be searched

        Returns:
            A list with the names of all variables that are candidates to be removed, but only one at a time.
        """
        if categoricals is None:
            categoricals = CategoryStruct.categoricals(data)

        ret = []
        if len(categoricals) < 2:
            return ret

        card_all_in = CategoryStruct.cardinality_set_of_rows(data, categoricals)

        for col in categoricals:
            others = [c for c in categoricals if c != col]
            card_others = CategoryStruct.cardinality_set_of_rows(data, others)

            if card_others == card_all_in:
                ret.append(col)

        return ret