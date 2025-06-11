import { useState, type DragEvent } from "react";
import axios from "axios";
import { v4 as uuidv4 } from "uuid";

const categories = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"];

export default function DragAndDrop() {
  const [selectedCategory, setSelectedCategory] = useState("");
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState("");
  const [imageId, setImageId] = useState("");
  const [uploadDate, setUploadDate] = useState("");
  const [metadata, setMetadata] = useState({ description: "", location: "" });
  const [showForm, setShowForm] = useState(false);
  const [imageFile, setImageFile] = useState<File | null>(null);

  const handleDrop = async (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setError("");
    setShowForm(false);

    const file = event.dataTransfer.files[0];
    if (!file || !file.type.startsWith("image/")) {
      setError("Please drop an image file.");
      return;
    }

    setImagePreview(URL.createObjectURL(file));
    setImageFile(file);
    setIsUploading(true);

    try {
      const formData = new FormData();
      formData.append("file", file);
      // Debug: inspect FormData
      for (const [key, value] of formData.entries()) {
        console.log(key, value);
      }

      const res = await axios.post("http://127.0.0.1:8000/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      if (res.data && categories.includes(res.data.prediction)) {
        setSelectedCategory(res.data.prediction);
        const newId = uuidv4();
        setImageId(newId);
        setUploadDate(new Date().toISOString());
        setShowForm(true);
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

  const handleSave = async () => {
    try {
      const payload = {
        id: imageId,
        uploadDate,
        category: selectedCategory,
        metadata,
        fileName: imageFile?.name,
      };

      await axios.post("/api/save", payload);
      alert("Saved to database!");
    } catch (err) {
      console.error(err);
      alert("Error saving to database.");
    }
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

      {showForm && (
        <div className="space-y-3 mt-4">
          <div>
            <label className="font-medium">Image ID:</label>
            <p className="bg-gray-100 p-2 rounded text-sm">{imageId}</p>
          </div>

          <div>
            <label className="font-medium">Upload Date:</label>
            <p className="bg-gray-100 p-2 rounded text-sm">{uploadDate}</p>
          </div>

          <div>
            <label className="block mb-1 font-medium">Predicted Category:</label>
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

          <div>
            <label className="block mb-1 font-medium">Description:</label>
            <input
              type="text"
              value={metadata.description}
              onChange={(e) => setMetadata({ ...metadata, description: e.target.value })}
              className="w-full border p-2 rounded"
              placeholder="Enter description"
            />
          </div>

          <div>
            <label className="block mb-1 font-medium">Location:</label>
            <input
              type="text"
              value={metadata.location}
              onChange={(e) => setMetadata({ ...metadata, location: e.target.value })}
              className="w-full border p-2 rounded"
              placeholder="Enter location"
            />
          </div>

          <button onClick={handleSave} className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700">
            Save to Database
          </button>
        </div>
      )}

      {/* <div className="mt-4">
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
      </div> */}
    </div>
  );
}
