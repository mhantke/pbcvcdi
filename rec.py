import spimage,numpy

def reconstruct(I,M,S):
    R = spimage.Reconstructor()
    R.set_intensities(I,shifted=False)
    R.set_mask(M,shifted=False)
    R.set_number_of_iterations(4000)
    R.set_number_of_outputs(100)
    R.set_initial_support(radius=10)
    R.set_phasing_algorithm("raar",beta_init=0.9,beta_final=0.9,constraints=["enforce_positivity"],number_of_iterations=2000)
    R.append_phasing_algorithm("raar",beta_init=0.9,beta_final=0.9,constraints=["enforce_positivity"],number_of_iterations=1000)
    R.append_phasing_algorithm("er",constraints=["enforce_positivity"],number_of_iterations=1000)
    R.set_support_algorithm("area",blur_init=6.,blur_final=6.,area_init=S["area_init_1st"],area_final=S["area_final_1st"],update_period=100,number_of_iterations=2000,center_image=False)
    R.append_support_algorithm("area",blur_init=6.,blur_final=1.,area_init=S["area_init_2nd"],area_final=S["area_final_2nd"],update_period=100,number_of_iterations=1000,center_image=False)
    R.append_support_algorithm("static",number_of_iterations=1000,center_image=False)
    O = R.reconstruct()
    return O
