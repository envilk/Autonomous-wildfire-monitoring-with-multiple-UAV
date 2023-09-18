import mesa
import agents
import random
import functools


class Fire(mesa.Agent):

    def __init__(self, unique_id, model, burning=False, fuel_bottom_limit=3, fuel_upper_limit=6, fire_spread_speed=2):
        super().__init__(unique_id, model)
        self.fuel = random.randint(fuel_bottom_limit, fuel_upper_limit)
        self.burning = burning
        self.next_burning_state = None
        self.moore = True
        self.radius = 2
        self.selected_dir = 0
        self.steps_counter = 0
        self.fire_spread_speed = fire_spread_speed

    def is_burning(self):
        return self.burning

    def get_fuel(self):
        return round(self.fuel)

    def probability_of_fire(self):
        probs = []
        if self.fuel > 0:
            adjacent_cells = self.model.grid.get_neighborhood(
                self.pos, moore=self.moore, include_center=False, radius=self.radius
            )
            for adjacent in adjacent_cells:
                agents_in_adjacent = self.model.grid.get_cell_list_contents([adjacent])
                for agent in agents_in_adjacent:
                    if type(agent) is agents.Fire:
                        adjacent_burning = 1 if agent.is_burning() else 0
                        probs.append(1 - (self.model.distance_rate(self.pos, adjacent, self.radius) * adjacent_burning))
            P = 1 - functools.reduce(lambda a, b: a * b, probs)
        else:
            P = 0
        return P

    def step(self):
        self.steps_counter += 1
        # make fire spread slower
        if self.steps_counter % self.fire_spread_speed == 0:
            prob_actual_cell = self.probability_of_fire()
            generated = random.random()
            if generated < prob_actual_cell:
                self.next_burning_state = True
            else:
                self.next_burning_state = False
            if self.burning and self.fuel > 0:
                self.fuel = self.fuel - self.model.burning_rate

    def advance(self):
        # make fire spread slower
        if self.steps_counter % self.fire_spread_speed == 0:
            self.burning = self.next_burning_state


class UAV(mesa.Agent):

    def __init__(self, unique_id, model, radius=2):
        super().__init__(unique_id, model)
        self.moore = True
        self.radius = radius
        self.selected_dir = 0

    def not_UAV_adjacent(self, pos):
        can_move = True
        agents_in_pos = self.model.grid.get_cell_list_contents([pos])
        for agent in agents_in_pos:
            if type(agent) is agents.UAV:
                can_move = False
        return can_move

    def surrounding_states(self):
        surrounding_states = []
        reward = 0
        positions = []
        surrounding_states = []
        adjacent_cells = self.model.grid.get_neighborhood(
            self.pos, moore=self.moore, include_center=True, radius=self.radius
        )
        for cell in adjacent_cells:
            agents = self.model.grid.get_cell_list_contents([cell])
            for agent in agents:
                if type(agent) is Fire:
                    surrounding_states.append(int(agent.is_burning() is True))
                    to_sum = 1 if agent.is_burning() else 0
                    reward += to_sum
                    positions.append(cell)
        return surrounding_states, reward, positions

    def move(self):
        # directions = [0, 1, 2, 3]  # right, down, left, up
        # self.selected_dir = random.choice(directions)
        move_x = [1, 0, -1, 0]
        move_y = [0, -1, 0, 1]

        pos_to_move = (self.pos[0] + move_x[self.selected_dir], self.pos[1] + move_y[self.selected_dir])
        if not self.model.grid.out_of_bounds(pos_to_move) and self.not_UAV_adjacent(pos_to_move):
            self.model.grid.move_agent(self, tuple(pos_to_move))

    def advance(self):
        self.move()
