import numpy as np

class reward_config():
    pdf = np.loadtxt("/home/ubuntu/obstates/project/data/pdf/td3-pdf.csv", delimiter=",")
    pdf_x = pdf[:,0]
    pdf_y = pdf[:,1]
    pdf_y = pdf_y / np.sum( pdf_y)
    
    ql = 0.2
    
    tick = 1e-8
    
def get_standard_state(bin_sizes, ob):
    rates = np.unique( ob )[1:]
    inds = [ np.arange(len(ob))[ ob == i].min() for i in rates]
    return np.concatenate( ( 10**rates, [0], bin_sizes[inds[:-1]] ))

def searchsorted( x, arr, side="right"):
    return min(np.searchsorted(arr, x, side=side), len(arr) - 1)

def get_reward(cls, *, action, state, mid_price):
    """"""

    evol = (
        lambda pv: np.array(
            [
                np.sum(np.minimum(np.maximum(cls.pdf_x - x, 0), cls.ql) * cls.pdf_y)
                for x in pv
            ]
        )
    )

    def evol_with_fill_sizes(pv):

        ev = []

        all_fill_sizes = []
        for x in pv:

            fill_sizes = np.minimum(np.maximum(cls.pdf_x - x, 0), cls.ql)

            all_fill_sizes.append([x, fill_sizes])

            ev.append(np.sum(fill_sizes * cls.pdf_y))

        return np.array(ev), all_fill_sizes

    """
    Round the action by converting to an actual rate, rounding,
    and then converting back again
    """
    # action = 1 - round(mid_price * (1 - action), cls.precision) / mid_price

    rates, pv = np.array_split(state, 2)

    adjusted_tick = cls.tick / mid_price

    rate_idx = searchsorted(action, rates)

    ev = evol(pv)

    expected_profit = action * ev[rate_idx]

    max_ep = np.max((rates - adjusted_tick) * ev)

    reward = expected_profit - max_ep



    return (reward.item(), expected_profit, max_ep)
