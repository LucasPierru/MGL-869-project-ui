import { useState, type DragEvent } from "react";
import axios from "axios";

const categories = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"];

export default function DragAndDrop() {
  const [selectedCategory, setSelectedCategory] = useState("");
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState("");

  const handleDrop = async (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setError("");
    const file = event.dataTransfer.files[0];
    if (!file || !file.type.startsWith("image/")) {
      setError("Please drop an image file.");
      return;
    }

    setImagePreview(URL.createObjectURL(file));
    setIsUploading(true);

    try {
      const formData = new FormData();
      formData.append("image", file);

      const res = await axios.post("/api/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      if (res.data && categories.includes(res.data.prediction)) {
        setSelectedCategory(res.data.prediction);
      } else {
        setError("Prediction not recognized.");
      }
    } catch (err) {
      console.error(err);
      setError("Error uploading or predicting image.");
    } finally {
      setIsUploading(false);
    }
  };

  const handleDragOver = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
  };

  return (
    <div className="max-w-md mx-auto mt-10 p-4 border rounded shadow">
      <div
        className="w-full px-16 h-40 border-dashed border-2 rounded flex items-center justify-center text-gray-500 bg-gray-50 hover:bg-gray-100"
        onDrop={handleDrop}
        onDragOver={handleDragOver}>
        {imagePreview ? (
          <img src={imagePreview} alt="Preview" className="max-h-full" />
        ) : (
          <p>Drag and drop an image here</p>
        )}
      </div>

      {isUploading && <p className="mt-2 text-blue-500">Uploading...</p>}
      {error && <p className="mt-2 text-red-500">{error}</p>}

      <div className="mt-4">
        <label className="block mb-1 font-medium">Select Category:</label>
        <select
          value={selectedCategory}
          onChange={(e) => setSelectedCategory(e.target.value)}
          className="w-full border p-2 rounded">
          <option value="">-- Choose one --</option>
          {categories.map((cat) => (
            <option key={cat} value={cat}>
              {cat[0].toUpperCase() + cat.slice(1)}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}
