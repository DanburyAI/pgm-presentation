import sys
import os
sys.path.append('./deps/pgmpy')
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model = BayesianModel([
    ('RatDebtToIncome', 'PaymentHist'),
    ('RatDebtToIncome', 'CreditWorthy'),
    ('Age', 'PaymentHist'),
    ('Age', 'Reliability'),
    ('Income', 'Assets'),
    ('Income', 'FutureIncome'),
    ('PaymentHist', 'Reliability'),
    ('Assets', 'FutureIncome'),
    ('Reliability', 'CreditWorthy'),
    ('FutureIncome', 'CreditWorthy')
])

"""
Low
High
"""
cpd_rat_debt_to_income = TabularCPD(
    variable='RatDebtToIncome', 
    variable_card=2, 
    values=[[0.5, 0.5]]
)

"""
16-21
22-64
over65
"""
cpd_age = TabularCPD(
    variable='Age', 
    variable_card=3, 
    values=[[1/3., 1/3., 1/3.]]
)

"""
High
Medium
Low
"""
cpd_income = TabularCPD(
    variable='Income', 
    variable_card=3, 
    values=[[1/3., 1/3., 1/3.]]
)

"""
Excellent
Acceptable
Unexeptable
"""
cpd_payment_hist = TabularCPD(
    variable='PaymentHist', 
    variable_card=3, 
    values=[[0.3,0.20,0.5,0.40,0.60,0.5],
            [0.2,0.15,0.3,0.25,0.25,0.3],
            [0.5,0.65,0.2,0.35,0.15,0.2]],
    evidence=['Age','RatDebtToIncome'],
    evidence_card=[3, 2]
)

"""
High
Medium
Low
"""
cpd_assets = TabularCPD(
    variable='Assets', 
    variable_card=3, 
    values=[[0.7,0.5,0.2],
            [0.2,0.3,0.4],
            [0.1,0.2,0.4]],
    evidence=['Income'],
    evidence_card=[3]
)

"""
Reliable
Unreliable
"""
cpd_reliability = TabularCPD(
    variable='Reliability', 
    variable_card=2, 
    values=[[0.65,0.7,0.75,0.45,0.55,0.65,0.3,0.4,0.5],
            [0.35,0.3,0.25,0.55,0.45,0.35,0.7,0.6,0.5]],
    evidence=['PaymentHist', 'Age'],
    evidence_card=[3,3]
)

"""
promising
not_promising
"""
cpd_future_income = TabularCPD(
    variable='FutureIncome', 
    variable_card=2, 
    values=[[0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1],
            [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]],
    evidence=['Assets', 'Income'],
    evidence_card=[3,3]
)

"""
positive
negative
"""
cpd_credit_worthy = TabularCPD(
    variable='CreditWorthy', 
    variable_card=2, 
    values=[[0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2],
            [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]],
    evidence=['RatDebtToIncome', 'Reliability', 'FutureIncome'],
    evidence_card=[2,2,2]
)

model.add_cpds(
    cpd_rat_debt_to_income,
    cpd_age,
    cpd_income, 
    cpd_payment_hist,
    cpd_assets,
    cpd_reliability, 
    cpd_future_income, 
    cpd_credit_worthy
)
assert(model.check_model())

# setup (exact) inference engine
infer = VariableElimination(model)

def task1():
    """using an observed RatDebtToIncome to predict age
    """
    print('initial probabilities - no observations')
    print(infer.query(['RatDebtToIncome'])['RatDebtToIncome'])
    print(infer.query(['Age'])['Age'])
    print('age probabilities - RatDebtToIncome observed')
    print(infer.query(['Age'], evidence={ 'RatDebtToIncome': 0 })['Age'])
    print('age probabilities - RatDebtToIncome,Reliability observed')
    print(infer.query(['Age'], evidence={ 'RatDebtToIncome': 0, 'Reliability': 1 })['Age'])

