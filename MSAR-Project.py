#!/usr/bin/env python
# coding: utf-8

# In[48]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def blackbody_spectrum(wavelengths, temperature):
    # Constants
    h = 6.62607015e-34  
    c = 2.99792458e8    
    k = 1.38064852e-23  

    wavelengths = wavelengths * 1e-9


    numerator = 2 * h * c ** 2
    exponent = (h * c) / (wavelengths * k * temperature)
    denominator = wavelengths ** 5 * ((np.exp(exponent) - 1))
    spectral_radiance = numerator / denominator

    
    spectrum = dict(zip(wavelengths, spectral_radiance))

    return spectrum

def plot_blackbody_spectrum(spectrum):
    
    
    wavelengths = np.array(list(spectrum.keys()))
    spectral_radiance = np.array(list(spectrum.values()))

    # Plot the spectrum
    plt.plot(wavelengths, spectral_radiance)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Spectral Radiance')
    plt.title('Blackbody Spectrum')
    plt.show()

# Example usage
wavelength_range = np.linspace(400, 780, 780)  # Wavelength range in nanometers
temperature = 5000  # Temperature in Kelvin

spectrum = blackbody_spectrum(wavelength_range, temperature)
plot_blackbody_spectrum(spectrum)


# In[49]:


dataset= blackbody_spectrum(wavelength_range, temperature)
dataset


# In[50]:


wavelength= list(dataset.keys())
#print(wavelength)


# In[51]:


spectral_radiance= list(dataset.values())


# In[ ]:





# In[52]:


df = pd.DataFrame(dataset.items(), columns=['wavelength', 'spectral_radiance'])
df = df.explode('spectral_radiance')
print(df)


# In[ ]:





# In[53]:


#df.to_csv('file_name.csv')


# In[54]:


#df1= pd.read_csv("file_name.csv")
#df1[:3]


# In[55]:


#import os

#with open("file_name.csv") as rf, open("file_name_org.csv", "w") as wf:
#    for i, line in enumerate(rf):
#        if i != 0:  # Everything but not the first line
#            wf.write(line)

#os.replace("file_name_org.csv", "file_name.csv")


# In[56]:


def read_sdss_spectrum(wavelength, flux):
    #wavelengths = df[:, 0]    # Wavelength values
    #flux = data[:, 2]           # Flux values

    # Normalize the flux values to obtain a probability distribution
    probabilities = flux / np.sum(flux)

    # Create a list of tuples to store the spectrum
    spectrum = list(zip(wavelength, probabilities))

    return spectrum


# In[57]:


output= read_sdss_spectrum(wavelength, spectral_radiance)
print(output)


# In[58]:


dataframe= pd.DataFrame(output, columns= ["wavelengths", "probabilities"])


# In[59]:


dataframe


# In[ ]:





# In[ ]:





# In[60]:


import random

def generate_simulated_photons(spectrum, num_photons):
    simulated_photons = []

    wavelengths, probabilities = zip(*spectrum)
    cumulative_probabilities = np.cumsum(probabilities)

    for _ in range(num_photons):
        random_num = random.random()
        wavelength = wavelengths[np.searchsorted(cumulative_probabilities, random_num)]
        energy = 1240 / wavelength  # Convert wavelength to energy in eV
        simulated_photons.append((wavelength, energy))

    return simulated_photons


# In[61]:


spectrum = [(500, 0.3), (600, 0.5), (700, 0.2)]
num_photons = 100

simulated_photons = generate_simulated_photons(spectrum, num_photons)
print(simulated_photons)


# In[62]:


import collections
counter = collections.Counter(simulated_photons)
counter


# In[ ]:





# In[ ]:





# In[66]:


import random
import numpy as np

def generate_simulated_photons(spectrum, num_photons):
    simulated_photons = []

    wavelengths, probabilities = zip(*spectrum)
    cumulative_probabilities = np.cumsum(probabilities)

    for _ in range(num_photons):
        random_num = random.random()
        wavelength = wavelengths[np.searchsorted(cumulative_probabilities, random_num)]
        energy = 1240 / wavelength  # Convert wavelength to energy in eV
        simulated_photons.append((wavelength, energy))

    return simulated_photons


def main():
    spectrum = [(500, 0.3), (600, 0.5), (700, 0.2)]  # Example spectrum
    num_photons = 100

    simulated_photons = generate_simulated_photons(spectrum, num_photons)
    print(simulated_photons)


if __name__ == '__main__':
    main()


# In[ ]::





# In[ ]:




