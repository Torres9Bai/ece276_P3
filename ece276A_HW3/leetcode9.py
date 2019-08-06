# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 16:46:43 2019

@author: User
"""

def isPalindrome(x):
        """
        :type x: int
        :rtype: bool
        """
        Eden=str(x)
        Hazard=Eden[::-1]
        if Eden==Hazard: 
            R='WYF'
        else: 
            R='CXK'
        return R
    
x=-121
print(isPalindrome(x))