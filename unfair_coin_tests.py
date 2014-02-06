#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
unfair_coin_tests.py

Some tests for the unfair_coin puzzle.

@author:  Luis Martin Gil
@contact: martingil.luis@gmail.com
@website: www.luismartingil.com
@github: https://github.com/luismartingil
'''

from unfair_coin import percentOutOfRange, Coin, HEADS, TAILS, processFairness
import unittest

class TestCoin(unittest.TestCase):

    def test_coin_invalid_range(self):
        """ Out of range coins must reigger percentOutOfRange """
        for p in [-0.5, -0.1, 1.1, 1.5]:
            self.assertRaises(percentOutOfRange, Coin, (p))

    def test_coin_throwing(self):
        """ Making sure the coins are prob valid """
        times = 50000
        alpha = times * 0.01
        for p in [0.5 + float(x)/10 for x in range(0, 5)]:
            coin = Coin(p)
            self.assertEqual(coin.getPercent(), p)
            simulation = {HEADS : 0, TAILS : 0}
            for _ in range(times):
                simulation[coin.throw()] += 1
            # We do have to check the probabilities
            self.assertTrue((simulation[HEADS] + alpha) >= (times * p))

class TestFairness(unittest.TestCase):
    def test_fairness_returns(self):
        """ Some basic testing for the fairness function """
        expected_dict = \
            {
            0 : 0.0,
            25 : 0.5,
            50 : 1.0,
            75 : 0.5,
            100 : 0.0,
            }
        for value, expected in expected_dict.iteritems():
            self.assertEqual(processFairness(value), expected)

if __name__ == '__main__':
    unittest.main()
