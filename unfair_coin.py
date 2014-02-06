#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
unfair_coin.py

Implementing the unfair coin puzzle.

@author:  Luis Martin Gil
@contact: martingil.luis@gmail.com
@website: www.luismartingil.com
@github: https://github.com/luismartingil
'''

import time
import random
import matplotlib
matplotlib.use('AGG') # Don't use a X-windows backend.
from matplotlib.pylab import figure,axes,title,savefig,close,subplots_adjust
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np

DPI_LIST=[60, 80, 100, 120, 150, 200, 300]
#P_LIST=[0.5]
P_LIST=[0.5, 0.6, 0.8]  # Probability of the coins to be used.
min_val=300 # Setting an exp distribution with a min_val
#N_LIST = range(min_val) + [(min_val + 2**x) for x in range(1, 10)]
N_LIST=range(min_val)
ACCURACY=500 # Iterations for each N to get an accuracy result.
FIGSIZE = (8,7)
#FIGSIZE = (7,4)

# enumerate
TAILS='tails'
HEADS='heads'

PLAY_TAILS='player-tails'
PLAY_HEADS='player-heads'

class percentOutOfRange(Exception):
    pass

class Coin(object):
    """ Simulates a coin game. """
    p = None
    def __init__(self, p):
        if 0.0 < p <= 1.0:
            self.p = float(p)
        else:
            msg = 'Given percent of the range. p:%s' % p
            raise percentOutOfRange(msg)
    def throw(self):
        """ Throws the coin to the air """
        return HEADS if (random.random() <= self.p) else TAILS
    def getPercent(self):
        return self.p
    def __str__(self):
        return '%s%s%% %s%s%%' % (HEADS, self.p * 100, TAILS, 100 - self.p * 100)

class Game(object):
    coin = None
    simulation = None
    def __init__(self, c):
        self.coin = c
    def simulate(self, n):
        self.simulation = {HEADS : 0, TAILS : 0}
        for i in range(n):
            self.simulation[self.coin.throw()] += 1
        return self.whoWon()
    def whoWon(self):
        if self.simulation:
            # PLAY_TAILS wins in case of even
            return self.simulation, PLAY_HEADS if self.getCondHeads() else PLAY_TAILS

class UnfairGame(Game):
    def getCondHeads(self):
        return self.simulation[HEADS] > self.simulation[TAILS]
    def __str__(self):
        return 'UnfairGame'
    def getColor(self):
        return 'r'

class FairGame(Game):
    def getCondHeads(self):
        # PLAY_HEADS needs to win more to win the game. Handicap ;-)
        # @luismartingil solution to the problem.
        total = self.simulation[HEADS] + self.simulation[TAILS]
        return (self.simulation[HEADS] > (self.coin.getPercent() * total))
    def __str__(self):
        return 'FairGame'
    def getColor(self):
        return 'g'

def processFairness(percent):
    """ Return a value in [0, 1] defining how fair is the percent.
    0, worst.
    1, best.
    
    Applies the fairness function based on:
    if x == 50 , y = 1
    if 0 <= x < 50, y = x/50.0
    if 50 < x <= 100, y=-x/50.0 + 2
    
    |
    |      (50,1)
    |         _
    |        /|\
    |       / | \
    |      /  |  \
    |     /   |   \
    |    /    |    \
    |   /     |     \
    |  /      |      \
    | /       |       \
    |/        |        \
    ---------------------------------------------------------------
    (0,0)   (50,0)   (100,0)
    
    """
    if (percent == 50): ret = 1.0
    elif (0 <= percent < 50): ret = float(percent) / 50.0
    elif (50 < percent <= 100): ret = (float(-percent) / 50.0) + 2.0
    else: raise percentOutOfRange('Error calculating fairness. percent:%s' % percent)
    return ret

class Result(object):
    texts = None
    values = None
    def __init__(self):
        self.texts = {'coin' : '%30s', 
                      'game' : '%12s', 
                      'n' : '%8s', 
                      'percent_player_heads_win' : '%24s', 
                      'percent_player_tails_win' : '%24s', 
                      'fairness_ratio' : '%20s'}
        self.values = {}

    def __str__(self):
        ret = []
        item = ''
        for t, format in self.texts.iteritems():
            item += (format % t)
        ret.append('#' + '~' * 60)
        for coin, game_dict in self.values.iteritems():
            for game, result_list in game_dict.iteritems():
                ret.append('# %30s %30s' % (coin, game))
                for v in result_list:
                    item = ' '
                    for t, format in self.texts.iteritems():
                        item += (format % v[t])
                    # end for
                    ret.append(item)
        return "\n".join(ret)

    def add(self, coin_key, game_key, **kwargs):
        """ Add another item to the internal result list """
        if not self.values.has_key(coin_key):
            self.values[coin_key] = {}
        if not self.values[coin_key].has_key(game_key):
            self.values[coin_key][game_key] = []
        self.values[coin_key][game_key].append(kwargs)

    def plot(self):
        """ Creates the plot in a png format based on the x and y keys """
        def filter(coin, game, key):
            """ DRY Helper function to make a list of a item from a list of dicts. """
            return [(value[key]) for value in self.values[coin][game]]

        font = {'family' : 'serif',
                'color'  : 'black',
                'weight' : 'normal',
                'size'   : 16,
                }

        size_leyend = 10
        coin_length = len(self.values)
        i = 0
        
        for coin, game_dict in sorted(self.values.iteritems()):
            i += 1
            for game, result_list in sorted(game_dict.iteritems()):
                title = '%s - coin:%s' % (i, coin)
                desc = game
                n_list = filter(coin, game, 'n')
                tails_list = filter(coin, game, 'percent_player_tails_win')
                heads_list = filter(coin, game, 'percent_player_heads_win')
                fairness_ratio_list = filter(coin, game, 'fairness_ratio')
                assert len(n_list) == len(tails_list) == len(heads_list)
                color = self.values[coin][game][0]['game'].getColor()
                self.make_heads_plot((coin_length, 1, i), FIGSIZE, font, size_leyend, title, desc, color, n_list, heads_list)
                #self.make_fairness_plot((coin_length, 1, i), FIGSIZE, font, size_leyend, title, desc, color, n_list, fairness_ratio_list)
        for dpi in DPI_LIST:
            savefig('player-heads-wins_%s.png' % dpi, bbox_inches='tight', dpi=dpi)
        close()

    def make_heads_plot(self, pos, figsize, font, size_leyend, title, desc, color, n_list, heads_list):
        """ Ugly make a plot-method. I know I didn't do my best. """
        plt.figure(1, figsize=figsize)
        subplots_adjust(bottom=0.065, left=0.05, right=0.975, hspace=0.6, wspace = 0.2)
        ax1 = plt.subplot(*pos)
        plt.title(title)
        #plt.plot(n_list, tails_list, marker='o', linestyle='-', color='r', label='win-tails', alpha=0.25)
        plt.plot(n_list, heads_list, marker='o', color=color, label=desc, alpha=0.25)
        plt.xlabel('n', fontdict=font)
        plt.ylabel('heads-wins', fontdict=font)
        plt.legend(loc='lower right', fancybox=True, prop={'size':size_leyend})
        #plt.grid(True)
        ax1.set_xlim([-1, len(n_list)])
        ax1.set_ylim([-10, 110])
        #ax1.set_xticklabels(())
        #start, end = ax1.get_xlim()
        #ax1.xaxis.set_ticks(np.arange(start, end, 1.712123))
        ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f%%'))
        plt.show()

    def make_fairness_plot(self, pos, figsize, font, size_leyend, title, desc, color, n_list, fairness_ratio_list):
        """ Ugly make a plot-method. I know I didn't do my best. """
        plt.figure(1, figsize=figsize)
        subplots_adjust(bottom=0.065, left=0.05, right=0.975, hspace=0.6, wspace = 0.2)
        ax2 = plt.subplot(*pos)
        plt.title(title)
        #plt.title('Fairness (%s)' % title, fontdict=font)
        plt.plot(n_list, fairness_ratio_list, marker='o', color=color, label=desc, alpha=0.25) #linestyle='-', 
        plt.xlabel('n', fontdict=font)
        plt.ylabel('fairness', fontdict=font)
        plt.legend(loc='lower right', fancybox=True, prop={'size':size_leyend})
        #plt.grid(True)
        ax2.set_xlim([-1, len(n_list)])
        ax2.set_ylim([-0.1, 1.2])
        #ax2.set_xticklabels(())
        plt.show()

if __name__ == '__main__':
    # Lets create a object where the result is going to be stored.
    # Little unefficient since storing the results in memory and then going through
    # all of them to plot.
    result = Result()
    for coin in [Coin(x) for x in P_LIST]:
        games = [FairGame(coin), UnfairGame(coin)] # Using the same unweighted coin for both
        for game in games:
            print '~' * 60
            tstart_game = time.time()        
            for n in N_LIST:
                tstart = time.time()
                result_game = {PLAY_HEADS : 0, PLAY_TAILS : 0}
                for acc in range(ACCURACY):
                    simulation, winner = game.simulate(n)
                    result_game[winner] += 1
                # end accuracy loop
                tfin = time.time()
                ttotal = float(tfin - tstart)
                percent_tmp = float(result_game[PLAY_HEADS] * 100) / float(acc) # removed round
                percent_player_heads_win = percent_tmp if (percent_tmp < 100.0) else 100.0
                percent_player_tails_win = 100.0 - percent_player_heads_win
                fairness_ratio = processFairness(percent_player_heads_win)
                result.add(str(coin),
                           str(game),
                           coin=coin,game=game,n=n,
                           percent_player_heads_win='%.5f' % percent_player_heads_win,
                           percent_player_tails_win='%.5f' % percent_player_tails_win,
                           fairness_ratio='%.3f' % fairness_ratio)
                print 'added: ', game, n, ttotal, '%.3f' % fairness_ratio, '%.5f' % percent_player_heads_win
            tfin_game = time.time()
            ttotal_game = float(tfin_game - tstart_game)
            print 'elapsed time: ', ttotal_game
    # Let's plot this!
    result.plot()
    print result
