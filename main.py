import mesa
import model
import agents
import time
from Canvas_Grid_Visualization import CanvasGrid

WIDTH = 60
HEIGHT = 60
VEGETATION_COLORS = ["Gray", "#9eff89", "#85e370", "#72d05c", "#62c14c", "#459f30", "#389023", "#2f831b",
                     "#236f11", "#1c630b", "#175808", "#124b05"]  # index is remaining fuel when IT ISN'T burning
FIRE_COLORS = ["Gray", "#d8d675", "#eae740", "#fefa01", "#fed401", "#feaa01", "#fe7001", "#fe5501",
               "#fe3e01", "#fe2f01", "#fe2301", "#fe0101"]  # index is remaining fuel when IT IS burning
FUEL_UPPER_LIMIT = 14
FUEL_BOTTOM_LIMIT = 11
COLORS_LEN = len(VEGETATION_COLORS)


# function that normalize fuel values to fit them with vegetation and fire colors
def normalize_fuel_values(fuel):
    return max(0, round((fuel / FUEL_UPPER_LIMIT) * COLORS_LEN - 1))


# creates agent dictionary for rendering it on Canvas Gird
def agent_portrayal(agent):
    portrayal = {"Shape": "rect", "Filled": True, "h": 1, "w": 1}
    if type(agent) is agents.Fire:
        if agent.is_burning():
            idx = normalize_fuel_values(agent.get_fuel())
            portrayal.update({"Color": FIRE_COLORS[idx], "Layer": 0})  # "#ff5d00" -> fire orange
        else:
            idx = normalize_fuel_values(agent.get_fuel())
            portrayal.update({"Color": VEGETATION_COLORS[idx], "Layer": 0})
    elif type(agent) is agents.UAV:
        portrayal.update({"Color": "Black", "Layer": 1, "h": 0.8, "w": 0.8})
    return portrayal


def main():
    # in python [height, width] for grid, in js [width, heigh]
    training = input('Type down for process kind \"train\" or \"eval\": ')
    RNN = input("Type down for predictor \"False\" or \"True\": ") == 'True'
    wf_model = model.WildFireModel(WIDTH, HEIGHT, False, FUEL_BOTTOM_LIMIT, FUEL_UPPER_LIMIT, RNN)
    if training == "train":
        wf_model.train()
    elif training == "eval":
        interface = input("Type down for evaluation \"Interface\" or \"Automatic\": ") == 'Interface'
        if interface:
            grid = CanvasGrid(agent_portrayal, WIDTH, HEIGHT, 10 * WIDTH, 10 * HEIGHT)
            # initialize Modular server for mesa Python visualization
            server = mesa.visualization.ModularServer(
                model.WildFireModel, [grid], "WildFire Model", {"width": WIDTH, "height": HEIGHT, "load": True,
                                                                "fuel_bottom_limit": FUEL_BOTTOM_LIMIT,
                                                                "fuel_upper_limit": FUEL_UPPER_LIMIT, "rnn": RNN}
            )
            server.port = 8521  # The default
            server.launch()
        else:
            print('Launched automatic evaluation, without interface ...')
            type = 'predictor' if RNN is True else 'entrenamiento'
            print(type)
            wf_model.evaluation(type)

main()
