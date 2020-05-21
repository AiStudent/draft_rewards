

def fak(n):
    if n==0:
        return 1
    return n*fak(n-1)


def probability(wins, losses, p):
    return (1-p) * (fak(wins+losses) / (fak(wins) * fak(losses))) * ((1-p)**losses) * (p**wins)


def xgames_probability(wins, losses, p):
    return (fak(wins+losses) / (fak(wins) * fak(losses))) * ((1-p)**losses) * (p**wins)

# gems gold packs
quick_draft = {   # 5000 gold 750 gems 3 losses
    0: (50, 0, 1.2),
    1: (100, 0, 1.22),
    2: (200, 0, 1.24),
    3: (300, 0, 1.26),
    4: (450, 0, 1.3),
    5: (650, 0, 1.35),
    6: (850, 0, 1.4),
    7: (950, 0, 2),
    'cost': 750,
    'win_max': 7,
    'losses': 3,
    'name': 'quick_draft'
}

premier_draft = { # 10000 gold 1500 gems 3 losses
    0: (50, 0, 1),
    1: (100, 0, 1),
    2: (250, 0, 2),
    3: (1000, 0, 2),
    4: (1400, 0, 3),
    5: (1600, 0, 4),
    6: (1800, 0, 5),
    7: (2200, 0, 6),
    'cost': 1500,
    'win_max': 7,
    'losses': 3,
    'name': 'premier_draft'
}

traditional_draft = {
    0: (0, 0, 1),
    1: (0, 0, 1),
    2: (1000, 0, 4),
    3: (3000, 0, 6),
    'cost': 1500,
    'games': 3,
    'name': 'traditional_draft'
}

standard_event = {
    0: (100, 3, 0),
    1: (200, 3, 0),
    2: (300, 3, 0),
    3: (400, 3, 0),
    4: (500, 3, 0),
    5: (600, 2, 1),
    6: (800, 1, 2),
    7: (1000, 1, 2),
    'cost': 500,
    'win_max': 7,
    'losses': 3,
    'name': 'standard_event'

}

traditional_event = {
    0: (0, 3, 0),
    1: (500, 3, 0),
    2: (1000, 3, 0),
    3: (1500, 2, 1),
    4: (1700, 2, 1),
    5: (2100, 1, 2),
    'cost': 1000,
    'win_max': 5,
    'losses': 2,
    'name': 'traditional_event'
}

import numpy as np
losses_available = 3
max_wins = 7

def check_avg(draft_rewards, win_prob=0.5):
    win_max = draft_rewards['win_max']
    probs = np.array([0 for _ in range(win_max+1)], dtype='float')

    #probs[0] = 5
    #print(probs)
    #quit()
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=2)

    exp_last = 0
    losses_available = draft_rewards['losses']
    for wins in range(0, win_max):

        p = probability(wins, losses_available-1, win_prob)
        exp_last += p
        #print(wins, p, p*np.array(draft_rewards[wins]))
        probs[wins] = p


    p = 1 - exp_last
    probs[win_max] = p
    #print(probs)
    #print(win_max, p, p * np.array(draft_rewards[win_max]))

    #print(probs)
    rewards = sum([probs[wins] * np.array(draft_rewards[wins]) for wins in range(0, win_max+1)])
    #rewards[2] *= 200
    #print(rewards / draft_rewards['cost'])
    #print(str(draft_rewards['name']).ljust(18),  rewards)
    return list(rewards)

def check_avg_x_games(draft_rewards, win_prob=0.5):
    probs = np.array([0,0,0,0], dtype='float')
    p = win_prob
    for wins in range(0,4):
        probs[wins] = xgames_probability(wins, 3-wins, p)

    #print(probs)
    rewards = sum([probs[wins] * np.array(draft_rewards[wins]) for wins in range(0, 4)])
    #print(str(draft_rewards['name']).ljust(18),  rewards)
    return list(rewards)





check_avg(quick_draft)  # ranked
check_avg(premier_draft)  # ranked
check_avg(standard_event)
check_avg(traditional_event)
check_avg_x_games(traditional_draft)     # unranked

import matplotlib.pyplot as plt


x = np.arange(0, 0.7, 0.01)
fig, ax1 = plt.subplots()
#ax2 = ax1.twinx()


drafts = [quick_draft, traditional_draft, premier_draft]

for picked_rares in range(0,4):
    for dc in [(quick_draft, 'blue'), (traditional_draft, 'green'), (premier_draft, 'salmon')]:
        draft, color = dc

        if draft is traditional_draft:
            arr = [check_avg_x_games(draft, xp) for xp in x]
        else:
            arr = [check_avg(draft, xp) for xp in x]

        gemscost = [-row[0]+draft['cost'] for row in arr]
        packs = [row[2] for row in arr]

        #print(draft['name'])
        #print(gemscost)
        #print(packs)

        # packs/gems lost
        # a pack is
        packspergamesspent = [ (packs[n] + picked_rares/1.15)*200/gemscost[n] for n in range(len(gemscost))]

        ax1.plot(x, packspergamesspent, color=color)
        ax1.set_xlabel('win prob')
        ax1.set_ylabel('packs/200gems spent (includes gem rewards)')
        #ax2.plot(x, packs, color='light' + color)
        #ax2.set_ylabel('packs')

baseline = ((0 * x) + 1 )

ax1.plot(x, baseline, color='red')
ax1.legend([draft['name'] for draft in drafts])
ax1.set(ylim=(0, 4))
plt.show()





