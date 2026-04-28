#!/usr/bin/env python3
"""
generate_map.py — Génère la carte PGM du monde hospital_full.sdf
Usage : python3 generate_map.py
Sortie : hospital_map.pgm (dans le dossier courant)

La carte correspond EXACTEMENT au layout SDF :
  Résolution : 0.05 m/pixel
  Origine    : (-1, -1) m
  Taille monde : 23 × 21 m  → 480 × 440 pixels
"""

import numpy as np
import struct
import os

# ── Paramètres carte ──────────────────────────────────────────────────────
RESOLUTION   = 0.05   # m / pixel
ORIGIN_X     = -1.0   # m
ORIGIN_Y     = -1.0   # m
WORLD_W      = 24.0   # m (X)
WORLD_H      = 22.0   # m (Y)

WIDTH  = int(WORLD_W / RESOLUTION)   # colonnes
HEIGHT = int(WORLD_H / RESOLUTION)   # lignes

FREE     = 254   # espace libre (blanc)
OCCUPIED = 0     # mur (noir)
UNKNOWN  = 205   # inconnu (gris)

# Grille initialisée à UNKNOWN
grid = np.full((HEIGHT, WIDTH), UNKNOWN, dtype=np.uint8)


def world_to_pixel(wx, wy):
    """Convertit coordonnées monde → pixel (col, row)."""
    col = int((wx - ORIGIN_X) / RESOLUTION)
    row = HEIGHT - 1 - int((wy - ORIGIN_Y) / RESOLUTION)
    return col, row


def fill_rect(x_min, y_min, x_max, y_max, value):
    """Remplit un rectangle (coordonnées monde) avec une valeur."""
    c1, r2 = world_to_pixel(x_min, y_min)
    c2, r1 = world_to_pixel(x_max, y_max)
    c1 = max(0, min(WIDTH-1,  c1))
    c2 = max(0, min(WIDTH-1,  c2))
    r1 = max(0, min(HEIGHT-1, r1))
    r2 = max(0, min(HEIGHT-1, r2))
    grid[r1:r2+1, c1:c2+1] = value


def draw_wall_h(x_min, x_max, y, thickness=0.2):
    fill_rect(x_min, y - thickness/2, x_max, y + thickness/2, OCCUPIED)


def draw_wall_v(x, y_min, y_max, thickness=0.2):
    fill_rect(x - thickness/2, y_min, x + thickness/2, y_max, OCCUPIED)


# ── Espaces libres (rooms + couloirs) ─────────────────────────────────────

# Pharmacie
fill_rect(0, 0, 10, 8, FREE)
# Couloir horizontal
fill_rect(0, 8, 22, 12, FREE)
# Couloir vertical
fill_rect(10, 0, 14, 20, FREE)
# Consultation
fill_rect(0, 12, 10, 20, FREE)
# Réanimation
fill_rect(14, 12, 22, 20, FREE)
# Urgences
fill_rect(14, 0, 22, 8, FREE)

# ── Murs extérieurs ───────────────────────────────────────────────────────
draw_wall_h(0, 22, 0,  0.2)   # Sud
draw_wall_h(0, 22, 20, 0.2)   # Nord
draw_wall_v(0,  0, 20, 0.2)   # Ouest
draw_wall_v(22, 0, 20, 0.2)   # Est

# ── Murs entre Pharmacie et couloir horizontal (y=8) ──────────────────────
# Ouverture porte : x=4.0→5.2
draw_wall_h(0, 4.0, 8)
draw_wall_h(5.2, 10, 8)

# ── Murs entre Consultation et couloir (y=12) ─────────────────────────────
draw_wall_h(0, 4.0, 12)
draw_wall_h(5.2, 10, 12)

# ── Mur entre couloir vertical et Réanimation (x=14, nord) ───────────────
# Ouverture porte : y=14.0→15.2
draw_wall_v(14, 12, 14.0)
draw_wall_v(14, 15.2, 20)

# ── Mur entre couloir vertical et Urgences (x=14, sud) ───────────────────
# Ouverture porte : y=4.8→6.0
draw_wall_v(14, 0, 4.8)
draw_wall_v(14, 6.0, 8)

# ── Mur entre Pharmacie et couloir vertical (x=10, bas) ──────────────────
# Ouverture porte : y=5.5→6.7
draw_wall_v(10, 0, 5.5)
draw_wall_v(10, 6.7, 8)

# ── Mur entre Consultation et couloir vertical (x=10, haut) ──────────────
# Ouverture porte : y=13.3→14.5
draw_wall_v(10, 12, 13.3)
draw_wall_v(10, 14.5, 20)

# ── Obstacles / mobilier (simplifié pour Nav2) ────────────────────────────

# Comptoir pharmacie (occupe de l'espace nav)
fill_rect(2.0, 6.2, 8.5, 7.0, OCCUPIED)

# Étagères pharmacie
fill_rect(0.85, 0.2, 1.15, 6.0, OCCUPIED)

# Lits réanimation
fill_rect(15.0, 16.8, 16.0, 19.2, OCCUPIED)
fill_rect(17.0, 16.8, 18.0, 19.2, OCCUPIED)
fill_rect(20.0, 16.8, 21.0, 19.2, OCCUPIED)

# Poste infirmier réanimation
fill_rect(16.5, 13.2, 19.5, 13.8, OCCUPIED)

# Lits urgences
fill_rect(15.0, 5.4, 16.0, 7.6, OCCUPIED)
fill_rect(17.0, 5.4, 18.0, 7.6, OCCUPIED)

# Chariot crash
fill_rect(20.7, 6.7, 21.3, 7.3, OCCUPIED)

# Bureaux consultation
fill_rect(1.8, 17.1, 3.2, 17.9, OCCUPIED)
fill_rect(5.3, 17.1, 6.7, 17.9, OCCUPIED)

# Chaises attente
fill_rect(2.7, 13.2, 6.3, 13.8, OCCUPIED)

# Poubelle couloir
fill_rect(13.3, 8.3, 13.7, 8.7, OCCUPIED)

# ── Inflation zone de sécurité autour des murs (gris clair) ──────────────
# Dilate OCCUPIED regions by 1 pixel to create soft border
from scipy.ndimage import binary_dilation
occupied_mask = (grid == OCCUPIED)
dilated = binary_dilation(occupied_mask, iterations=2)
# Zone autour des murs reste inconnue/grise pour forcer Nav2 à l'éviter
grid[(dilated) & (~occupied_mask)] = UNKNOWN

# ── Écriture PGM ─────────────────────────────────────────────────────────
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "maps", "hospital_map.pgm")
os.makedirs(os.path.dirname(out_path), exist_ok=True)

with open(out_path, 'wb') as f:
    # En-tête PGM P5
    header = f"P5\n{WIDTH} {HEIGHT}\n255\n"
    f.write(header.encode('ascii'))
    f.write(grid.tobytes())

print(f"✅ Carte générée : {out_path}")
print(f"   Taille : {WIDTH} × {HEIGHT} pixels")
print(f"   Résolution : {RESOLUTION} m/pixel")
print(f"   Couverture : {WORLD_W} × {WORLD_H} m")
