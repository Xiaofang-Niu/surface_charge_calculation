# This script can be used for calculation of surface charge of a given crystal face and allows the screening of surface atoms within a specific distance.

# Created 04/10/2025 by Xiaofang Niu, Yanyang Li and Xiaobin Sun

from ccdc import io
from ccdc.particle import Surface
import numpy as np
import csv

# Input the crystal structure file
file_name = input("File name of the crystal structure: ")    
crystal_reader = io.CrystalReader(file_name)
analyzed_crystal = crystal_reader[0]

# Select the calculated surface
inputed_h = int(input("Select surface orientation h: "))
inputed_k = int(input("Select surface orientation k: "))
inputed_l = int(input("Select surface orientation l: "))
inputed_offset = int(input("Select offset (o): "))
inputed_u = int(input("Size of surface u: "))
inputed_v = int(input("Size of surface v: "))
depth = float(input("Select filtered depth: "))     # Can input non-integer depth

surface_001 = Surface(analyzed_crystal, (inputed_h, inputed_k, inputed_l), thickness_factor=1, offset=inputed_offset, surface_size=(inputed_u,inputed_v))
surface_atoms = surface_001.surface_atoms     # Acquire surface atoms

# The molecules on the surface are acquired and partial charges are assigned
atom_charge_dict = {}
for molecule in surface_001.slab_molecules:
    success = molecule.assign_partial_charges()
    if not success:
        print("Partial charge assignment failed for one of the molecules.")
        continue  
    for atom in molecule.atoms:
        atom_charge_dict[tuple(atom.coordinates)] = atom.partial_charge     # Each atom is associated with its partial charge by its unique coordinate

# The total surface charge was calculated using the absolute values of the atom charges and normalized by surface area or projected area
total_charge = sum(abs(atom_charge_dict[tuple(atom.coordinates)]) for atom in surface_atoms)
surface_area = surface_001.descriptors.surface_area
projected_area = surface_001.descriptors.projected_area
average_charge_per_surface_area = total_charge / surface_area if surface_area > 0 else float('inf')
average_charge_per_projected_area = total_charge / projected_area if projected_area > 0 else float('inf')

# Output results of surface atoms
print(f"Number of atoms in the area of {inputed_u} × {inputed_v}: {len(surface_atoms)}")     
print(f"Surface area of {inputed_u} × {inputed_v}: {surface_area} Å^2")
print(f"Projected area of {inputed_u} × {inputed_v}: {projected_area} Å^2")
print(f"Total surface charge: {total_charge}")
print(f"Surface charge per surface area: {average_charge_per_surface_area} Å^-2")
print(f"Surface charge per projected area: {average_charge_per_projected_area} Å^-2")

# Filter surface atoms within a specific distance
vector_c = np.array(tuple(surface_001.periodic_vectors[2]))
max_projection_c = max(np.dot(np.array(atom.coordinates), vector_c) / np.linalg.norm(vector_c) for atom in surface_atoms)
filtered_atoms_within_depth = []
for atom in surface_atoms:
    atom_position = np.array(atom.coordinates)
    projection_c = np.dot(atom_position, vector_c) / np.linalg.norm(vector_c)
    if abs(projection_c - max_projection_c) <= depth:
        charge = atom_charge_dict.get(tuple(atom.coordinates), 'N/A')
        if isinstance(charge, (int, float)):
            filtered_atoms_within_depth.append((atom, charge))

# The total surface charge of filtered surface atoms was calculated using the absolute values of the atom charges and normalized by surface area or projected area
total_charge_filtered = sum(abs(charge) for _, charge in filtered_atoms_within_depth)
average_charge_per_surface_area_filtered = total_charge_filtered / surface_area if surface_area > 0 else float('inf')
average_charge_per_projected_area_filtered = total_charge_filtered / projected_area if projected_area > 0 else float('inf')
 
# Output results of filtered surface atoms
print(f"Number of atoms within {depth} Å of the highest point atom: {len(filtered_atoms_within_depth)}")    
print(f"Total surface charge of filtered surface atoms: {total_charge_filtered}")
print(f"Surface charge per surface area (filtered): {average_charge_per_surface_area_filtered} Å^-2")
print(f"Surface charge per projected area (filtered): {average_charge_per_projected_area_filtered} Å^-2")
 
# Write the surface charge information to a csv file
csv_file_name = input("Input the file name of the surface charge information: ")
with open(csv_file_name, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Atom Label", "Charge"])
    for atom, charge in filtered_atoms_within_depth:
        writer.writerow([atom.label, charge])
print(f"The surface charge information of filtered atoms has been written: {csv_file_name}")
