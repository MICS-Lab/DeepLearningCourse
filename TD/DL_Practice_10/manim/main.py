from manim import *
from tracker_updater import text_updater


class GanEvolutionThroughEpochs(Scene):
    def construct(self):

        # Time list
        time_list = np.linspace(0, 18.49*3600, 31)  # 18h29

        # Paths stuff
        root_dir = "../label_smoothing/images_without_skip_with_double_res_block_act_no_act"

        list_images: list[ImageMobject] = []
        arrows: tuple[CurvedArrow, CurvedArrow]
        for k in range(0, 301, 10):
            list_images.append(
                ImageMobject(
                    root_dir + f"/{k}/val_0.png"
                ).move_to(ORIGIN).scale(1.25)
            )
        width_image = list_images[-1].get_width()
        arrow_1_2: CurvedArrow = CurvedArrow(
            start_point=list_images[0].get_top()-[7/16*width_image, 0, 0],
            end_point=list_images[0].get_top()-[5/16*width_image, 0, 0],
            angle=-PI / 2.5,  # Modify the angle to change the curvature of the arrow
            tip_length=0.2,  # Modify the tip_length to make the arrow tip smaller
        )
        arrow_2_3: CurvedArrow = CurvedArrow(
            start_point=list_images[0].get_top()-[5/16*width_image, 0, 0],
            end_point=list_images[0].get_top()-[3/16*width_image, 0, 0],
            angle=-PI / 2.5,  # Modify the angle to change the curvature of the arrow
            tip_length=0.2,  # Modify the tip_length to make the arrow tip smaller
        )
        arrows = (
            arrow_1_2,
            arrow_2_3,
        )

        # Add text labels
        gen_no_glasses = Text("gen_no_glasses").next_to(arrow_1_2, UP, buff=0.1).scale(0.25 ).shift(DOWN*0.25)
        gen_glasses = Text("gen_glasses").next_to(arrow_2_3, UP, buff=0.1).scale(0.25).shift(DOWN*0.25)
        title = Text("CycleGAN evolution through epochs").next_to(list_images[0], DOWN).scale(0.8)

        # MObjects
        Text.set_default(font_size=20)

        tracker = ValueTracker(0)
        temps_position = np.array([2.5, 2.5, 0])
        temps_de_base = Tex("Time spent training the GAN :", " 0.00 s").scale(0.8)
        temps_de_base.shift(temps_position-temps_de_base[0].get_center())
        last_image = None
        self.play(Write(temps_de_base), runtime=0.005)


        # On met les images générées
        for idx, image in enumerate(list_images):

            temps_de_base.add_updater(lambda text: text_updater(text, tracker, temps_position))

            if last_image:
                self.play(
                    AnimationGroup(
                        FadeOut(last_image),
                        FadeIn(image),
                        tracker.animate.set_value(time_list[idx]),
                    ),
                    runtime=0.005
                )
            else:
                self.play(
                    AnimationGroup(
                        FadeIn(image),
                        Write(arrows[0]),
                        Write(arrows[1]),
                        Write(gen_no_glasses),
                        Write(gen_glasses),
                        Write(title),
                        tracker.animate.set_value(time_list[idx]),
                    ),
                    runtime=0.1
                )

            for updater in temps_de_base.updaters:
                temps_de_base.remove_updater(updater)

            self.wait(0.5)

            last_image = image
