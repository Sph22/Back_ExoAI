# Calculadora de parámetros astronómicos para modelo de clasificación de exoplanetas.
# Solicita datos estelares/planetarios, calcula period, duration, depth, radius, insolation, teff, srad.
# Cada fórmula empleada está explicada en los comentarios.

import math
import csv
import os

# Definimos constantes físicas y de conversión
G_cgs   = 6.6743e-8        # Constante gravitacional en unidades cgs (cm^3 g^-1 s^-2)
M_sun_g = 1.98847e33       # Masa solar en gramos
R_sun_cm = 6.957e10        # Radio solar en cm
# Conversión de radio solar a UA y de radio solar a radios terrestres:
R_sun_AU = 1.0 / 215.032   # 1 R☉ = 1/215.032 UA ≈ 0.00465 UA (aprox)
R_sun_to_Rearth = 109.1    # 1 R☉ ≈ 109.1 R⊕ (radios de la Tierra)

print("Ingrese los siguientes datos. Deje la entrada vacía y presione Enter si desconoce un valor.\n")

# 1. Masa estelar (Masa solar)
masa_str = input("Masa estelar (en masas solares) [dejar vacío si no se conoce]: ")
if masa_str.strip() == "":
    M_star = None
else:
    # Reemplazamos coma por punto por si el usuario usa coma decimal
    masa_str = masa_str.replace(",", ".")
    try:
        M_star = float(masa_str)
    except ValueError:
        M_star = None

# 2. Semieje mayor de la órbita (UA)
a_str = input("Semieje mayor de la órbita (en UA) [vacío si no se conoce]: ")
if a_str.strip() == "":
    a_AU = None
else:
    a_str = a_str.replace(",", ".")
    try:
        a_AU = float(a_str)
    except ValueError:
        a_AU = None

# 3. Radio estelar (R_sun)
rstar_str = input("Radio de la estrella (en radios solares) [vacío si no]: ")
if rstar_str.strip() == "":
    R_star = None
else:
    rstar_str = rstar_str.replace(",", ".")
    try:
        R_star = float(rstar_str)
    except ValueError:
        R_star = None

# 4. Temperatura efectiva de la estrella (K)
teff_str = input("Temperatura efectiva de la estrella (en K) [vacío si no]: ")
if teff_str.strip() == "":
    T_eff = None
else:
    teff_str = teff_str.replace(",", ".")
    try:
        T_eff = float(teff_str)
    except ValueError:
        T_eff = None

# 5. log(g) de la estrella (cm/s^2, log10)
logg_str = input("log(g) de la estrella [vacío si no]: ")
if logg_str.strip() == "":
    log_g = None
else:
    logg_str = logg_str.replace(",", ".")
    try:
        log_g = float(logg_str)
    except ValueError:
        log_g = None

# 6. Luminosidad estelar (en luminosidades solares)
lum_str = input("Luminosidad de la estrella (en L_sun) [vacío si no]: ")
if lum_str.strip() == "":
    L_star = None
else:
    lum_str = lum_str.replace(",", ".")
    try:
        L_star = float(lum_str)
    except ValueError:
        L_star = None

# 7. Profundidad del tránsito (en ppm, p.ej. 1500) 
depth_str = input("Profundidad del tránsito (en ppm) [vacío si no]: ")
if depth_str.strip() == "":
    depth_ppm_input = None
else:
    depth_str = depth_str.replace(",", ".")
    try:
        depth_ppm_input = float(depth_str)
    except ValueError:
        depth_ppm_input = None

# 8. Duración observada del tránsito (horas)
dur_str = input("Duración observada del tránsito (en horas) [vacío si no]: ")
if dur_str.strip() == "":
    transit_dur_obs = None
else:
    dur_str = dur_str.replace(",", ".")
    try:
        transit_dur_obs = float(dur_str)
    except ValueError:
        transit_dur_obs = None

# (Opcional) Otros datos podrían añadirse aquí si fueran necesarios en el futuro.

print("\nCalculando parámetros...")

# Preparar variables de salida
period_days    = None
duration_hours = None
depth_ppm_out  = None
radius_rearth  = None
insolation     = None
teff_out       = None
srad_out       = None

# Calcular radio estelar si falta, usando logg y M* o usando luminosidad y Teff
if R_star is None:
    if (M_star is not None) and (log_g is not None):
        # Calcula R* a partir de logg: g = G * M / R^2  ->  R = sqrt(G M / g)
        g_cgs = 10**log_g            # gravedad en cm/s^2
        M_star_g = M_star * M_sun_g  # masa estelar en gramos
        R_star_cm = math.sqrt(G_cgs * M_star_g / g_cgs)  # resultado en cm
        R_star = R_star_cm / R_sun_cm  # convertir a radios solares
        # Documentación: se usó g = GM/R^2, con G en cgs.
    elif (L_star is not None) and (T_eff is not None):
        # Si conocemos L* y Teff, usar Stefan-Boltzmann: L = (R^2)*(T_eff/T_sun)^4
        # => R = R_sun * sqrt( L_star / (T_eff/T_sun)^4 )
        T_sun = 5778.0  # temperatura efectiva del Sol en K (aprox)
        R_star = math.sqrt( L_star / ((T_eff / T_sun)**4) )
        # Aquí R_star queda en unidades de R_sun, dado L_star en L_sun y T_eff en K.
        # (L_star/L_sun) = (R_star/R_sun)^2 * (T_eff/T_sun)^4  -> despejado arriba.
    # Si ninguna opción aplica, R_star seguirá siendo None.

# Calcular luminosidad estelar si falta, usando R* y Teff
if L_star is None:
    if (R_star is not None) and (T_eff is not None):
        # L_star/L_sun = (R_star/R_sun)^2 * (T_eff/T_sun)^4
        T_sun = 5778.0
        L_star = (R_star**2) * ((T_eff / T_sun)**4)
        # L_star se obtiene en unidades de L_sun.

# Calcular temperatura efectiva si falta, usando L* y R*
if T_eff is None:
    if (L_star is not None) and (R_star is not None):
        # T_eff = T_sun * (L_star / (R_star^2))^(1/4)
        T_sun = 5778.0
        T_eff = T_sun * ((L_star) / (R_star**2))**0.25

# Calcular período orbital (period) si es posible
if (a_AU is not None) and (M_star is not None):
    # Tercera ley de Kepler: P (años) = sqrt(a^3 / M_star). Convertir a días multiplicando por 365.25.
    period_years = math.sqrt((a_AU**3) / M_star)
    period_days = period_years * 365.25
    # Comentario: Suponemos masa del planeta << masa estelar.

# Calcular insolation (S) si es posible
if (L_star is not None) and (a_AU is not None):
    # Insolación relativa a la Tierra: S = (L_star/L_sun) / (a_AU^2)
    insolation = L_star / (a_AU**2)

# Calcular profundidad de tránsito (depth) de salida y radio planetario
if depth_ppm_input is not None:
    # Si el usuario proporcionó la profundidad en ppm, úsala directamente para depth_out.
    depth_ppm_out = depth_ppm_input
    if (R_star is not None):
        # Calcular radio planetario a partir de la profundidad y R*.
        depth_frac = depth_ppm_input / 1e6  # convertir ppm a fracción
        Rp_Rstar = math.sqrt(depth_frac)    # Rp/R* (adimensional)
        # Convertir Rp a radios de la Tierra: Rp = Rp_Rstar * R_star (en R_sun) * (R_sun en R_earth)
        radius_rearth = Rp_Rstar * R_star * R_sun_to_Rearth
else:
    # Si no se proporcionó profundidad pero sí radio planetario (no hay input explícito de Rp en nuestro caso),
    # podríamos calcular depth si supiéramos Rp. Como no se pide Rp como input, omitimos esta rama.
    pass

# Calcular profundidad de tránsito a partir de radio planetario (si algún día se añade Rp como input)
# [Este bloque se deja como referencia por si se extendiera la funcionalidad]
# if (depth_ppm_input is None) and (radius_rearth_input is not None) and (R_star is not None):
#     # Calcular profundidad desde Rp conocido
#     Rp_Rstar = (radius_rearth_input / R_sun_to_Rearth) / R_star  # convierte Rp (R_earth) a R_sun y luego divide por R_star
#     depth_frac = (Rp_Rstar ** 2)
#     depth_ppm_out = depth_frac * 1e6
#     radius_rearth = radius_rearth_input  # usamos el radio proporcionado como salida también

# Calcular duración del tránsito (duration) si no se proporcionó y tenemos datos suficientes
if transit_dur_obs is not None:
    # Si el usuario ya tiene la duración observada, la usamos directamente.
    duration_hours = transit_dur_obs
else:
    # Si no hay duración observada, intentar calcularla:
    if (period_days is not None) and (R_star is not None) and (a_AU is not None):
        # Usar fórmula aproximada de tránsito central: T = P * R_star / (pi * a)
        # Convertir R_star a unidades de semieje (UA) antes de aplicar fórmula.
        R_star_AU = R_star * R_sun_AU  # convertir R_star de R_sun a UA
        T_days = (period_days * R_star_AU) / (math.pi * a_AU)
        duration_hours = T_days * 24.0
        # Nota: Esta es una duración estimada (asumiendo tránsito por el centro).

# Preparar los valores de salida en el orden requerido: period, duration, depth, radius, insolation, teff, srad
# Usar None o "" para los que no se pudieron calcular.
period_out    = period_days
duration_out  = duration_hours
depth_out     = depth_ppm_out
radius_out    = radius_rearth
insolation_out= insolation
teff_out      = T_eff   # T_eff recalculada o dada por usuario
srad_out      = R_star  # R_star recalculado o dado por usuario

# Mostrar resultados por pantalla
print("\nResultados calculados:")
if period_out is not None:
    print(f"Periodo orbital (días): {period_out:.3f}")
else:
    print("Periodo orbital (días): No calculado (faltan datos)")

if duration_out is not None:
    print(f"Duración del tránsito (horas): {duration_out:.3f}")
else:
    print("Duración del tránsito (horas): No calculado")

if depth_out is not None:
    # Mostrar profundidad con un decimal si no es entero grande
    if abs(depth_out - round(depth_out)) < 1e-6:
        # Si es prácticamente entero
        print(f"Profundidad del tránsito (ppm): {int(round(depth_out))}")
    else:
        print(f"Profundidad del tránsito (ppm): {depth_out:.1f}")
else:
    print("Profundidad del tránsito (ppm): No calculado")

if radius_out is not None:
    print(f"Radio planetario (radios Tierra): {radius_out:.3f}")
else:
    print("Radio planetario (R⊕): No calculado")

if insolation_out is not None:
    print(f"Insolación relativa (S⊕): {insolation_out:.3f}")
else:
    print("Insolación relativa (S⊕): No calculado")

if teff_out is not None:
    # Redondear Teff al entero más cercano
    print(f"Temperatura efectiva estelar (K): {int(round(teff_out))}")
else:
    print("Temperatura efectiva estelar (K): No calculado")

if srad_out is not None:
    print(f"Radio estelar (R☉): {srad_out:.3f}")
else:
    print("Radio estelar (R☉): No calculado")

# Guardar resultados en CSV
output_values = []
for val in [period_out, duration_out, depth_out, radius_out, insolation_out, teff_out, srad_out]:
    if val is None:
        output_values.append("")  # dejar vacío si no calculado
    else:
        # Formatear el valor a texto (usar notación estándar)
        if isinstance(val, float):
            # Convertir a string de forma segura (evitar notación científica excesiva)
            # Limitar a 6 cifras significativas para evitar muchos decimales innecesarios
            output_values.append(f"{val:.6g}")
        else:
            output_values.append(str(val))

# Escribir en archivo CSV (append mode)
csv_filename = "calculated_inputs.csv"
try:
    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Si el archivo estaba vacío o no existía, podríamos opcionalmente escribir encabezados.
        # En este caso no se solicitan encabezados, solo agregamos la fila de resultados.
        writer.writerow(output_values)
    print(f"\nValores guardados en '{csv_filename}' (orden: period, duration, depth, radius, insolation, teff, srad).")
except Exception as e:
    print(f"\n**Error al escribir en {csv_filename}:**", e)