#!/bin/bash

# Limpieza total de procesos previos
killall -9 gzserver gzclient mavros sim_vehicle.py 2>/dev/null

echo "Iniciando Centro de Control TFG (8 Paneles)..."

# Lanzamos terminator con el layout tfg_drones
terminator -l tfg_drones &
