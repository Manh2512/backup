"""
Physics module for unsupervised model.
"""
import torch
import torch.fft as fft

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
latent_dim = 512
pixel_size = 2*10.2e-6
objective_mag = 60
z = 1e-3
wavelengths = torch.tensor(np.loadtxt('iSCAT_data/iscat_wavelengths.csv', delimiter=','), dtype=torch.float32)

def physics_module(reconstructed_image, pixel_size, wavelength, objective_mag, z):
    """
    Simulates the iSCAT forward propagation given an initial reconstructed image.

    Parameters:
    - reconstructed_image: torch tensor, the initial reconstructed image, of shape [batch, 1, 512, 512]
    - pixel_size: float, physical pixel size of the image (in meters)
    - wavelength: float, wavelength of the light used (in meters)
    - objective_mag: float, magnification of the objective lens
    - z: float, propagation distance (in meters)

    Returns:
    - iscat_image: torch tensor, the simulated iSCAT microscope image
    """
    
    # Constants
    k = (2 * 3.14159265 / (wavelength*1e-9))  # Wavenumber
    
    # Image size and pixel size adjustment for magnification
    image_size = reconstructed_image.shape
    dx = pixel_size / objective_mag  # Effective pixel size after magnification

    # Create the spatial coordinates (physical space)
    x = torch.linspace(-image_size[-1] // 2, image_size[-1] // 2 - 1, image_size[-1], dtype=torch.float32) * dx
    y = torch.linspace(-image_size[-2] // 2, image_size[-2] // 2 - 1, image_size[-2], dtype=torch.float32) * dx
    X, Y = torch.meshgrid(x, y, indexing='ij')  # Use 'ij' indexing to match NumPy behavior

    # Fresnel propagation kernel
    H = torch.exp(1j * k * z) * torch.exp(-1j * k / (2 * z) * (X**2 + Y**2))

    # Simulate the scattered field
    scattered_field = reconstructed_image * torch.exp(1j * k * z).to(device)

    # Apply Fourier transforms for propagation
    scattered_field_ft = fft.fftshift(fft.fft2(scattered_field))  # Forward FFT
    scattered_field_propagated = fft.ifft2(fft.ifftshift(scattered_field_ft * H))  # Apply kernel and inverse FFT

    # Reference beam (plane wave)
    reference_amplitude = 1.0
    reference_phase = torch.tensor(0.0)
    reference_field = reference_amplitude * torch.exp(1j * reference_phase) * torch.ones_like(scattered_field_propagated)

    # Compute iSCAT interference pattern
    iscat_image = torch.abs(scattered_field_propagated + reference_field)**2

    # Normalize the output
    iscat_image /= torch.max(iscat_image.detach())

    return iscat_image
