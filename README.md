# Déploiement FastAPI sur Vercel (Serverless)

Ce dossier expose l'API FastAPI en fonction serverless Vercel.

## Fichiers
- api/index.py : point d’entrée ASGI, réutilise l’app de deploy_render/backend_app/main.py
- requirements.txt : dépendances Python
- vercel.json : configuration Vercel (runtime, fichiers inclus, routes, env)
- data/ : place ici habitat_index_H.nc pour la démo (ou configure MAZI_DATA_PATH)

## Étapes
1. Ajouter ce dossier au repo GitHub
2. Sur Vercel : New Project → Import repository
3. Root directory : projet (vercel lit deploy_vercel/vercel.json)
4. Variables d’environnement (si besoin) :
   - MAZI_DATA_PATH (défaut: deploy_vercel/data/habitat_index_H.nc)
   - CORS_ALLOW_ORIGINS (URL de ton frontend)
5. Déployer

## Test local (vite fait)
- `pip install -r deploy_vercel/requirements.txt`
- Lancer via `uvicorn deploy_render.backend_app.main:app --reload --port 8000`
- (Sur Vercel, c’est serverless ; ce lancement local simule juste l’ASGI)
