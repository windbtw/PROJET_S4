import statmorph as statmorph
import create_galaxy as cg
import galaxy_functions as gf

galaxy = cg.generate_galaxy(amplitude=1, r_eff=20, n=2.5, x_0=120.5, y_0=96.5, ellip=0.5, theta=-0.5)
image = gf.load_image("test.jpg")
image1 = gf.load_image("pikachu.jpg")

print(gf.concentration_index(image1))
print(gf.compute_asymmetry(image1))
print(cg.plot_galaxy(image1))
#print(statmorph.SourceMorphology.concentration(galaxy))                   #marche pas jsp pourquoi