mkdir cropped
for f in *posterior-F1*pdf; do
   pdf-crop-margins -s -u -o "cropped/$f" "$f"
done
