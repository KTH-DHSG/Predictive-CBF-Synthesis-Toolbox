"""

    This module defines the CBF class, which is used to store and handle Control Barrier Functions (CBFs) on a grid. 

    The CBF class allows for the creation of a grid based on specified domain bounds and discretization parameters. It provides methods to compute grid points, extract unique grid points, and create an interpolator for the CBF values. The class also includes functionality to save and load CBF data to and from files in JSON format.
    The CBF values can be stored in a meshgrid format, and the class provides methods to handle both single numpy arrays and lists of numpy arrays. The save method creates a folder if it does not exist and saves the CBF data to a specified file, while the load method retrieves the data from a file and populates the CBF instance with the loaded values.

    The class is used is used for saving and storing computed CBFs and in the Controller class to compute safe control inputs.

    (c) Adrian Wiltz, 2025

"""

import os
import numpy as np
import Auxiliaries.auxiliary as aux
from scipy.interpolate import RegularGridInterpolator
import json
import functools
import operator
class CBF:
    """Class for storing values of a CBF on a grid. The grid is defined by a domain and a discretization."""
    
    def __init__(self,
                 domain_lower_bound=None,
                 domain_upper_bound=None,
                 discretization=None,
                 domain=None):
        """Constructor creates a uniform grid from domain bounds using the given discretization. 
        None of the entries of the points on the grid exceed the specified bounds. The grid points are computed by 
        incrementing the entries starting with the values specified in domian_lower_bound.

        Args:
            domain_lower_bound (NumPy array of length x_dim): lower bound of domain
            domain_upper_bound (NumPy array of length x_dim): upper bound of domain
            discretization (int array of length x_dim): number of grid points in respective dimension
            domain (NumPy meshgrid): custom domain
        """
        if domain_lower_bound is not None and domain_upper_bound is not None and discretization is not None:

            # compute domain as a meshgrid
            self.domain = self.computeGridPointsAsMeshgrid(domain_lower_bound,domain_upper_bound,discretization)
            
            self.cbf_values = np.empty_like(self.domain[0])

        elif domain:
            # constructor initializes CBF with a provided custom domain
            self.domain = domain
            
            self.cbf_values = np.empty_like(self.domain)

        else:
            # empty initialization of instance, can be used e.g. for loading data from a file
            pass

    def getPointList(self):
        """
        Generate a list of points in the domain along with their corresponding indices.
        This method creates a list of points from the domain (meshgrid) and pairs each point 
        with its corresponding index. The resulting list contains dictionaries, each with a 
        "point" key for the point coordinates and an "index" key for the point's index in the 
        domain.
        Returns:
            list: A list of dictionaries, each containing:
                - "point" (tuple): The coordinates of the point in the domain.
                - "index" (tuple): The index of the point in the domain.
        """
        
        # Create a list for the points in the domain (meshgrid), a list of the corresponding indices, and a so far empty list of the results
        points = list(zip(*[self.domain[i].flatten() for i in range(len(self.domain))]))
        indices_tmp = np.indices(self.domain[0].shape) # Create index arrays for each dimension
        indices = list(zip(*[indices_tmp[i].flatten() for i in range(len(indices_tmp))])) # Flatten and zip the indices to get a list of tuples

        # Pair each point with its corresponding index and a placeholder for the result
        point_list = [{"point": point, "index": index} for point, index in zip(points, indices)]
        
        return point_list
    
    def getCbfGridPoints(self):
        """
        Extracts unique grid points from the domain.

        This method iterates over each array in the `domain` attribute, flattens it,
        finds the unique values, and collects them into a tuple. The output can be used to generate an interpolator for the CBF.

        Returns:
            tuple: A tuple containing arrays of unique grid points from each array in the domain.
        """
        
        grid_point_tuple = ()
        for grid_array in self.domain:
            grid_point_tuple = grid_point_tuple + (np.unique(grid_array.flatten()),)

        return grid_point_tuple
    
    def getCbfInterpolator(self, method='linear'):
        """
        Create a RegularGridInterpolator for the CBF values.

        This method creates a RegularGridInterpolator object for the CBF values using the grid points from the domain and the CBF values.

        Returns:
            RegularGridInterpolator: A RegularGridInterpolator object for the CBF values.
        """
        
        # Get the unique grid points from the domain
        grid_points = self.getCbfGridPoints()
        
        # Create the interpolator
        interpolator = RegularGridInterpolator(grid_points,self.cbf_values,method=method,bounds_error=False,fill_value=None)
        
        return interpolator
        

    def save(self, filename, folder_name="Data"):
        # Create the folder if it does not exist
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            print(f"Folder '{folder_name}' created.")

        # Save the CBFmodule to a file
        print("Saving CBF to file started")

        attributes = {}
        for key, value in self.__dict__.items():
            print("Currently saving: ", key)
            if isinstance(value, np.ndarray):  # Handle single numpy arrays
                attributes[key] = {'ndarray': True, 'array': value.tolist()}
            elif isinstance(value, list) and all(isinstance(v, np.ndarray) for v in value):  # Handle lists of numpy arrays
                if aux.is_meshgrid(value):  # Check if it's a meshgrid (N-dimensional)
                    attributes[key] = {'meshgrid': True, 'arrays': [v.tolist() for v in value]}
                else:  # If it's just a list of arrays, save them normally
                    attributes[key] = {'ndarray': True, 'arrays': [v.tolist() for v in value]}
            else:  # Handle other data types directly
                attributes[key] = value

        file_path = os.path.join(folder_name, filename)

        print("Writing CBF data to file")
        with open(file_path, "w") as file:
            json.dump(attributes, file, indent=4)

        print("Saving CBF to file finished. \nNumer of data points saved: " + str(functools.reduce(operator.mul, self.cbf_values.shape, 1)))

    def load(self, filename, folder_name="Data"):
        # Create the file path
        file_path = os.path.join(folder_name, filename)

        with open(file_path, "r") as file:
            attributes = json.load(file)

        for key, value in attributes.items():
            if isinstance(value, dict):
                if value.get('meshgrid', False):  # Handle meshgrids
                    # Convert each array in the meshgrid back to numpy array
                    self.__dict__[key] = [np.array(arr) for arr in value['arrays']]
                elif value.get('ndarray', False):  # Handle single or list of arrays
                    if 'array' in value:
                        self.__dict__[key] = np.array(value['array'])
                    else:
                        self.__dict__[key] = [np.array(arr) for arr in value['arrays']]
            else:  # Handle other data types
                self.__dict__[key] = value

    @staticmethod
    def computeGridPointsAsMeshgrid(lower_bound, upper_bound, discretization):
        """
        This method computes a domain as a grid for a given lower and upper bound as well as an discretization. The domain is created by incrementing the entries starting with the values specified in the lower bound.

        Args:
            lower_bound (NumPy array of length x_dim): lower bound of domain
            upper_bound (NumPy array of length x_dim): upper bound of domain
            discretization (int array of length x_dim): number of grid points in respective dimension

        Returns:
            NumPy meshgrid: A meshgrid representing the domain.
        """
        # compute domain as a meshgrid
        args = ()
        
        for k in range(len(discretization)):
            args = args + (np.linspace(lower_bound[k],upper_bound[k],discretization[k]),)
        
        domain = np.meshgrid(*args)

        return domain
    
    @staticmethod
    def computeGridPoints(lower_bound, upper_bound, discretization):
        """
        For the specification of unique grid points in each dimension, this method computes a domain as a grid for a given lower and upper bound as well as an discretization. The domain is created by incrementing the entries starting with the values specified in the lower bound.

        Args:
            lower_bound (NumPy array of length x_dim): lower bound of domain
            upper_bound (NumPy array of length x_dim): upper bound of domain
            discretization (int array of length x_dim): number of grid points in respective dimension

        Returns:
            tuple: A tuple containing arrays of unique grid points from each array in the domain.
        """
        # compute domain as a meshgrid
        domain = CBF.computeGridPointsAsMeshgrid(lower_bound, upper_bound, discretization)
        
        grid_point_tuple = ()
        for grid_array in domain:
            grid_point_tuple = grid_point_tuple + (np.unique(grid_array.flatten()),)

        return grid_point_tuple