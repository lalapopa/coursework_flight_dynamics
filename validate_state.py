import os
from src.data.constants import Constant as cn


class Validator:
    def __init__(self, variant, path, pgf_state, png_state):
        self.variant = variant
        self.path = str(path)
        self.pgf_state = pgf_state
        self.png_state = png_state

    def variant_validate(self):
        try:
            self.variant.get()
        except:
            print("heello unvalid btw")
            return False

        try:
            cn(int(self.variant.get()), "png", "D:/")
            return True
        except Exception as e:
            return False

    def path_validate(self):
        if os.path.exists(self.path):
            return True
        return False

    def pgf_state_validate(self):
        if self.pgf_state:
            return True
        return False

    def png_state_validate(self):
        if self.png_state:
            return True
        return False
