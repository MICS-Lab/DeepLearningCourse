from manim import *


def text_updater(text, tracker, target_position):
    text.become(
        Tex("Time spent training the GAN : ", '{0:.2f}'.format(tracker.get_value()), " s").scale(0.8)
        # Recreate the text every frame
    )
    text.shift(
        target_position - text[0].get_center()
    )  # Shift it so "Temps passé : " stays motionless
