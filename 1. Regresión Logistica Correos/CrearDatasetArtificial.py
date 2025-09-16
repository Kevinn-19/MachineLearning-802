import pandas as pd
import random
import numpy as np
from datetime import datetime, timedelta

# ---------------------------
# Configuración inicial
# ---------------------------
n = 1000
n_spam = int(n * 0.50) # 50% SPAM 
n_ham = n - n_spam # 50% HAM

# Dominios
dominios_comunes = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com", "aol.com", "protonmail.com"]
dominios_corporativos = ["microsoft.com", "amazon.com", "google.com", "apple.com", "facebook.com", "tesla.com", "ibm.com"]
dominios_fakes = [
    "freemail.xyz", "cheapoffers.biz", "clicknow.info", "promo-mail.net", "lottery-win.org",
    "spamworld.co", "getrichfast.ru", "prizefree.cn", "offerx.io", "phishmail.in",
    "randomfake.site", "nowfree.win", "trustme.click"
]
dominios = dominios_comunes + dominios_corporativos + dominios_fakes

# Eventos (más probabilidad "Ninguno")
eventos = ["Navidad", "San Valentín", "Black Friday", "Año Nuevo", "Ninguno"]
prob_eventos = [0.05, 0.05, 0.05, 0.05, 0.8]

# Asuntos
asuntos_ham = [
    "Confirmación de tu compra", "Recordatorio de reunión a las 10 AM", "Informe mensual adjunto",
    "Actualización de términos y condiciones", "Factura electrónica disponible",
    "Invitación al evento corporativo", "Resumen de actividades semanales",
    "Tu pedido ha sido enviado", "Nueva actualización de seguridad disponible",
    "Notificación de inicio de sesión", "Felices fiestas de parte de nuestro equipo",
    "Cambio de contraseña solicitado", "Boletín de novedades de la empresa",
    "Detalles de tu reservación", "Recibo de pago electrónico"
]

asuntos_spam = [
    "Gana $10,000 ahora mismo", "Última oportunidad para reclamar tu premio",
    "Tu cuenta bancaria ha sido bloqueada", "Haz clic aquí para recibir tu herencia",
    "Descubre cómo bajar 10kg en una semana", "Oferta exclusiva solo para ti",
    "Entra ya y recibe tu regalo", "Confirma tus datos para no perder acceso",
    "Invierte hoy y duplica tu dinero", "Felicidades, eres el ganador",
    "Haz clic para obtener tu cupón gratis", "Consigue seguidores en minutos",
    "Accede a contenido exclusivo para adultos", "Compra medicamentos sin receta aquí",
    "Increíble promoción válida por tiempo limitado"
]

# ---------------------------
# Funciones auxiliares
# ---------------------------
def generar_fecha():
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 12, 31)
    delta = end_date - start_date
    random_days = random.randrange(delta.days)
    random_seconds = random.randrange(86400)
    fecha = start_date + timedelta(days=random_days, seconds=random_seconds)
    return fecha.strftime("%Y-%m-%d %H:%M:%S")

def codificar_dominio(dominio):
    if dominio in dominios_comunes:
        return 1  # Gratuito
    elif dominio in dominios_corporativos:
        return 2  # Empresa
    else:
        return 3  # Fake

def codificar_evento(evento):
    mapa = {"Navidad": 1, "San Valentín": 2, "Black Friday": 3, "Año Nuevo": 4, "Ninguno": 5}
    return mapa[evento]

def categorizar_hora(fecha_str):
    hora = pd.to_datetime(fecha_str).hour
    if 6 <= hora < 12:
        return 1  # Mañana
    elif 12 <= hora < 18:
        return 2  # Tarde
    elif 18 <= hora < 24:
        return 3  # Noche
    else:
        return 4  # Madrugada

def categorizar_asunto(asunto):
    asunto = asunto.lower()
    if any(pal in asunto for pal in ["compra", "factura", "pago", "pedido"]):
        return 1  # Compra/Facturación
    elif any(pal in asunto for pal in ["seguridad", "contraseña", "acceso", "inicio de sesión"]):
        return 2  # Seguridad/Acceso
    elif any(pal in asunto for pal in ["evento", "reunión", "invitación", "recordatorio"]):
        return 3  # Evento/Notificación
    elif any(pal in asunto for pal in ["oferta", "promoción", "premio", "dinero", "gratis", "ganar"]):
        return 4  # Promoción/Oferta
    else:
        return 5  # Otros

# ---------------------------
# Generar dataset
# ---------------------------
data = []

prob_ruido_ham = 0.15
prob_ruido_spam = 0.1

# SPAM
for _ in range(n_spam):
    dominio = random.choice(dominios)
    evento = random.choices(eventos, weights=prob_eventos, k=1)[0]
    
    # ¿Este SPAM parece HAM? 
    ruido = np.random.rand() < prob_ruido_spam
    
    fila = {
        "FechaHoraEnvio": generar_fecha(),
        "Asunto": random.choice(asuntos_spam),
        "DominioRemitente": codificar_dominio(dominio),
        "PalabrasPremios": 0 if ruido else random.choice([0, 1]),
        "TerminosFinancieros": 0 if ruido else random.choice([0, 1]),
        "CantidadDestinatarios": random.randint(1, 5) if ruido else random.randint(5, 50),
        "TieneEmojis": 0 if ruido else random.choice([0, 1]),
        "PalabrasMalEscritas": 0 if ruido else random.choice([0, 1]),
        "CantidadEnlaces": random.randint(0, 2) if ruido else random.randint(2, 10),
        "EnlacesAcortados": 0 if ruido else random.choice([0, 1]),
        "EventosEspecificos": codificar_evento(evento),
        "TipoCorreo": 1
    }
    data.append(fila)

# HAM
for _ in range(n_ham):
    dominio = random.choice(dominios)
    evento = random.choices(eventos, weights=prob_eventos, k=1)[0]
    
    # ¿Este HAM parece SPAM? 
    ruido = np.random.rand() < prob_ruido_ham
    
    fila = {
        "FechaHoraEnvio": generar_fecha(),
        "Asunto": random.choice(asuntos_ham),
        "DominioRemitente": codificar_dominio(dominio),
        "PalabrasPremios": 1 if ruido else 0,
        "TerminosFinancieros": 1 if ruido else random.choices([0, 1], weights=[0.8, 0.2])[0],
        "CantidadDestinatarios": random.randint(10, 50) if ruido else random.randint(1, 5),
        "TieneEmojis": 1 if ruido else random.choices([0, 1], weights=[0.9, 0.1])[0],
        "PalabrasMalEscritas": 1 if ruido else random.choices([0, 1], weights=[0.9, 0.1])[0],
        "CantidadEnlaces": random.randint(5, 10) if ruido else random.randint(0, 2),
        "EnlacesAcortados": 1 if ruido else 0,
        "EventosEspecificos": codificar_evento(evento),
        "TipoCorreo": 0
    }
    data.append(fila)

# Crear DataFrame
df = pd.DataFrame(data)

# Mezclar
df = df.sample(frac=1).reset_index(drop=True)

# ---------------------------
# Transformar columnas
# ---------------------------
df["FechaHoraEnvio"] = df["FechaHoraEnvio"].apply(categorizar_hora)
df["Asunto"] = df["Asunto"].apply(categorizar_asunto)

# Guardar CSV
df.to_csv("Regresión Logistica/dataset_correos_final.csv", index=False, encoding="utf-8")

# ---------------------------
# Resultados
# ---------------------------
print("Dataset final generado con transformaciones incluidas")
print(df["TipoCorreo"].value_counts(normalize=True))
print(df.head())
