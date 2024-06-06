'''
|============================================= ls.py ==============================================|
|                                                                                                  |
|   Property of Daniel Sturdivant. Unauthorized copying of this file via any medium would be       |
|   super sad and unfortunate for me. Proprietary and confidential.                                |
|                                                                                                  |
| ------------------------------------------------------------------------------------------------ |
|                                                                                                  |
|   @file     charlizard/algorithms/ls.py                                                          |
|   @brief    Class for least squares estimation.                                                  |
|   @author   Daniel Sturdivant <sturdivant20@gmail.com>                                           |
|   @date     February 2024                                                                        |
|                                                                                                  |
|==================================================================================================|
'''

import numpy as np

class LeastSquares:
  @property
  def estimate(self):
    return self._x_hat
  
  @property
  def measurement(self):
    return self._y_hat
  
  @property
  def observation_matrix(self):
    return self._H
  
  # === __INIT__ ===
  def __init__(self, meas_model):
    """constructor

    Parameters
    ----------
    meas_model : function
        Function defining measurement model. 
          Should receive:
            1. x_hat  -> initial state estimate
            2. params -> keyword arguments that define y=Hx relationship
          Should return:
            1. y_hat  -> estimate of the measurements
            2. H      -> observation matrix of the measurements
    """
    self._meas_model = meas_model
    self._x_hat = None
    self._y_hat = None
    self._H = None
  
  
  # === SET_INITIAL_CONDITIONS ===
  def set_initial_conditions(self, x_hat: np.ndarray):
    """provide initial conditions for iterative least squares

    Parameters
    ----------
    x_hat : np.ndarray
        Nx1 initial state estimate
    """
    self._x_hat = x_hat
  
  
  # === SOLVE ===
  def solve(self, method: str='linear', params: dict=None, y_tilde: np.ndarray=None, x_hat: np.ndarray=None) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    method : str, optional
        least squares type, by default 'linear'
            1. 'linear'     -> ordinary least squares
            2. 'iterative'  -> nonlinear least squares
            3. 'recursive'  -> static least squares
    params : dict, optional
        parameter dictionary for observable generation, by default None
    y_tilde : np.ndarray, optional
        actual sensor measurement, by default None
    x_hat : np.ndarray, optional
        initial conditions for iterative least squares, by default None

    Returns
    -------
    np.ndarray
        new least squares estimate

    Raises
    ------
    ValueError
        No initial conditions
    ValueError
        Incorrect method input
    """
    if (x_hat is None) and (self._x_hat is None) and (method.lower() == 'iterative'):
      print("LeastSquares::solve error. No initial conditions, must set 'x_hat'!")
      raise ValueError
    elif (x_hat is not None):
      self._x_hat = x_hat
    
    match method:
      case 'linear':
        self.__linear_solver(y_tilde, params)
      case 'iterative':
        self.__iterative_solver(y_tilde, params)
      # case 'recursive':
      #   self._x_hat, self._y_hat, self._H = _recursive_solver(params)
      case _:
        print("LeastSquares::solve error. Incorrect method input, must be 'linear', 'iterative', or 'recursive'!")
        raise ValueError
    
    return self._x_hat
  
  
  #! === _LINEAR_SOLVER ===
  # solve linear least squares problem
  def __linear_solver(self, y, params):
    self._y_hat, self._H = self._meas_model(y, **params)
    if 'weights' in params:
      # weighted
      w = np.diag(params['weights'])
      self._x_hat = np.linalg.inv(self._H.T @ w @ self._H) @ self._H.T @ w @ self._y_hat
    else:
      # unweighted
      self._x_hat = np.linalg.inv(self._H.T @ self._H) @ self._H.T @ self._y_hat
      

  #! === _ITERATIVE_SOLVER ===
  # solve nonlinear least squares problem
  def __iterative_solver(self, y, params):
    # update threshold
    t = 1e-6
    dx = 1e6
    if 'threshold' in params:
      t = params['threshold']
    
    if 'weights' in params:
      # weighted
      w = np.diag(params['weights'])
      while dx > t:
        self._y_hat, self._H = self._meas_model(self._x_hat, **params)
        x_old = self._x_hat
        delta = np.linalg.inv(self._H.T @ w @ self._H) @ self._H.T @ w @ (y - self._y_hat)
        self._x_hat += delta
        dx = np.linalg.norm(self._x_hat - x_old)      
    else:
      # unweighted
      i = 0
      while dx > t:
        self._y_hat, self._H = self._meas_model(self._x_hat, **params)
        x_old = self._x_hat
        delta = np.linalg.inv(self._H.T @ self._H) @ self._H.T @ (y - self._y_hat)
        self._x_hat += delta
        dx = np.linalg.norm(delta)
        i += 1
        