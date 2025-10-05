from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from typing import Optional
import os
import io
import datetime as dt
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


app = FastAPI(title="MaziShark API", version="0.1.0")

# Enable CORS (configurable via env CORS_ALLOW_ORIGINS="https://site1,https://site2")
cors_env = os.getenv("CORS_ALLOW_ORIGINS", "*")
allow_origins = [o.strip() for o in cors_env.split(",") if o.strip()] if cors_env else ["*"]
app.add_middleware(
	CORSMiddleware,
	allow_origins=allow_origins,
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


DATA_FILENAME = "habitat_index_H.nc"


def find_data_file() -> Optional[str]:
	"""Return absolute path to habitat_index_H.nc if present in workspace."""
	# 1) Env override
	override = os.getenv("MAZI_DATA_PATH")
	if override and os.path.exists(override):
		return os.path.abspath(override)
	# 2) Look in CWD, project root, backend, and data/
	candidates = [
		os.path.abspath(DATA_FILENAME),
		os.path.abspath(os.path.join(os.getcwd(), DATA_FILENAME)),
		os.path.abspath(os.path.join(os.path.dirname(__file__), "..", DATA_FILENAME)),
		os.path.abspath(os.path.join(os.path.dirname(__file__), DATA_FILENAME)),
		os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", DATA_FILENAME)),
		os.path.abspath(os.path.join("data", DATA_FILENAME)),
	]
	for p in candidates:
		if os.path.exists(p):
			return p
	return None


def load_h_dataset(path: str) -> xr.Dataset:
	if not os.path.exists(path):
		raise FileNotFoundError(f"Fichier introuvable: {path}")
	ds = xr.open_dataset(path)
	if "H_index" not in ds:
		raise KeyError("Variable 'H_index' absente du NetCDF")
	# Sanity: ensure coords lat/lon names
	if "lat" not in ds.coords or "lon" not in ds.coords:
		raise KeyError("Coordonnées 'lat'/'lon' absentes du NetCDF")
	return ds


@app.get("/meta")
def meta():
	"""Retourne metadonnées: taille lat/lon et bornes min/max."""
	data_path = find_data_file()
	if data_path is None:
		raise HTTPException(status_code=404, detail="Fichier H introuvable. Exécute le notebook.")
	ds = load_h_dataset(data_path)
	lat = ds["lat"].values
	lon = ds["lon"].values
	return {
		"path": data_path,
		"lat": {"size": int(lat.size), "min": float(lat.min()), "max": float(lat.max())},
		"lon": {"size": int(lon.size), "min": float(lon.min()), "max": float(lon.max())},
	}


@app.get("/health")
def health():
	return {"status": "ok"}


@app.get("/analyze")
def analyze(date: Optional[str] = Query(None, description="YYYY-MM-DD")):
	"""
	MVP: utilise le NetCDF déjà produit par le notebook.
	- Si présent: retourne stats de H et la date demandée (si fournie).
	- Sinon: explique comment générer H via le notebook.
	"""
	data_path = find_data_file()
	if data_path is None:
		return JSONResponse(
			status_code=200,
			content={
				"message": "Aucun fichier habitat_index_H.nc trouvé. Exécute le notebook pour le générer.",
				"expected_file": DATA_FILENAME,
			},
		)

	try:
		ds = load_h_dataset(data_path)
		H = ds["H_index"]
		hmin = float(np.nanmin(H.values))
		hmax = float(np.nanmax(H.values))
		hmean = float(np.nanmean(H.values))
		payload = {
			"data_path": data_path,
			"stats": {"min": hmin, "max": hmax, "mean": hmean},
		}
		if date is not None:
			# validate date
			try:
				dt.datetime.strptime(date, "%Y-%m-%d")
			except Exception:
				raise HTTPException(status_code=400, detail="Format de date invalide, attendu YYYY-MM-DD")
			payload["requested_date"] = date
		return payload
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Erreur d'analyse: {e}")


@app.get("/map")
def map_png():
	"""Retourne une carte PNG de H si disponible."""
	data_path = find_data_file()
	if data_path is None:
		raise HTTPException(status_code=404, detail="Fichier H introuvable. Exécute le notebook.")
	try:
		ds = load_h_dataset(data_path)
		H = ds["H_index"]
		lat = ds["lat"].values
		lon = ds["lon"].values
		# Render simple pcolormesh (PlateCarree-like)
		fig, ax = plt.subplots(figsize=(10, 6))
		im = ax.pcolormesh(lon, lat, H.values, cmap="viridis", shading="auto")
		ax.set_title("Indice d'habitat probable H(x,y)")
		ax.set_xlabel("Longitude")
		ax.set_ylabel("Latitude")
		plt.colorbar(im, ax=ax, label="H (0-1)")
		buf = io.BytesIO()
		plt.tight_layout()
		fig.savefig(buf, format="png", dpi=150)
		plt.close(fig)
		buf.seek(0)
		# Save to a temp file on disk to stream easily (Windows safe)
		tmp_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "map.png"))
		with open(tmp_path, "wb") as f:
			f.write(buf.read())
		return FileResponse(tmp_path, media_type="image/png", filename="map.png")
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Erreur génération carte: {e}")


@app.get("/plot")
def plot_png():
	"""MVP: retourne la même carte que /map (placeholder pour graphiques par couche)."""
	return map_png()


@app.get("/predict")
def predict(lat: float = Query(...), lon: float = Query(...)):
	"""
	Retourne H au voisin le plus proche (MVP). Si H absent, renvoie 0.5.
	"""
	data_path = find_data_file()
	if data_path is None:
		return {"lat": lat, "lon": lon, "H": 0.5, "note": "H absent, renvoi par défaut"}
	try:
		ds = load_h_dataset(data_path)
		H = ds["H_index"]
		# Nearest index
		lat_arr = ds["lat"].values
		lon_arr = ds["lon"].values
		i = int(np.clip(np.abs(lat_arr - lat).argmin(), 0, lat_arr.size - 1))
		j = int(np.clip(np.abs(lon_arr - lon).argmin(), 0, lon_arr.size - 1))
		val = float(H.values[i, j]) if np.isfinite(H.values[i, j]) else float("nan")
		return {"lat": lat, "lon": lon, "H": val, "i": i, "j": j}
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Erreur prédiction: {e}")


@app.get("/series")
def series(agg: str = Query("global", description="global|lat_mean|lon_mean")):
    """
    Retourne une série 1D dérivée de H pour graphiques (MVP):
    - global: histogramme grossier (10 bacs) des valeurs H
    - lat_mean: moyenne par latitude
    - lon_mean: moyenne par longitude
    """
    data_path = find_data_file()
    if data_path is None:
        raise HTTPException(status_code=404, detail="Fichier H introuvable. Exécute le notebook.")
    try:
        ds = load_h_dataset(data_path)
        H = ds["H_index"]
        if agg == "lat_mean":
            vals = np.nanmean(H.values, axis=1)
            lat = ds["lat"].values.tolist()
            return {"type": "lat_mean", "lat": lat, "H": [None if not np.isfinite(v) else float(v) for v in vals]}
        if agg == "lon_mean":
            vals = np.nanmean(H.values, axis=0)
            lon = ds["lon"].values.tolist()
            return {"type": "lon_mean", "lon": lon, "H": [None if not np.isfinite(v) else float(v) for v in vals]}
        # default: histogram
        arr = H.values.flatten()
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {"type": "global", "bins": [], "counts": []}
        counts, edges = np.histogram(arr, bins=10, range=(0.0, 1.0))
        bins = edges[:-1].tolist()
        return {"type": "global", "bins": bins, "counts": counts.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur séries: {e}")


