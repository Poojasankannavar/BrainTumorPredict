# Deploying the trained model to Render

The three exported files are required at runtime:

- `Best_BrainTumor_Model.pkl`
- `Scaler.pkl`
- `LabelEncoder.pkl`

They are already exported by cell 26 of `ml.ipynb`. The SVM model is about 256 MB, which is larger than GitHub's 100 MB normal Git limit. This repository is configured to store `.pkl` files with Git LFS instead.

Before pushing, run these commands from the repository root:

```powershell
git lfs install
git add .gitattributes .gitignore Best_BrainTumor_Model.pkl Scaler.pkl LabelEncoder.pkl app.py
git commit -m "Deploy trained brain tumor model with Git LFS"
git push origin main
```

Then open the Render service's **Manual Deploy** menu and select **Clear build cache & deploy**. Verify the build output checks out the LFS objects; the service must contain all three files beside `app.py` before it starts.

The service uses an ephemeral filesystem, so uploaded MRI images are only retained long enough to display the result. They are not permanent uploads.
