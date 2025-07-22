"use client";

import { useState } from "react";
import axios from "axios";
import FlashcardCarousel from "../components/FlashcardCarousel";
import "./globals.css";

export default function Home() {
  const [file, setFile] = useState();
  const [flashcards, setFlashcards] = useState([]);
  const [loading, setLoading] = useState(false);

  const uploadPDF = async () => {
    if (!file) return;
    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);
    const res = await axios.post(
      "http://localhost:8000/generate_flashcards",
      formData
    );
    setFlashcards(res.data.flashcards);
    setLoading(false);
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-sky-100 via-purple-100 to-pink-100 p-6 flex flex-col items-center justify-center">
      <h1 className="text-4xl font-bold mb-4 text-gray-800">
        📘 PDF to Flashcards
      </h1>
      <div className="bg-white shadow-lg rounded-2xl p-6 w-full max-w-md flex flex-col gap-4 items-center">
        <input
          type="file"
          accept="application/pdf"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          className="file:mr-4 file:px-4 file:py-2 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100  w-full"
        />
        <button
          onClick={uploadPDF}
          disabled={loading || !file}
          className="bg-indigo-600 text-white px-6 py-2 rounded-full hover:bg-indigo-700 transition cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? "Processing..." : "Upload & Generate"}
        </button>
      </div>

      {flashcards.length > 0 && (
        <div className="mt-10 w-full max-w-2xl">
          <FlashcardCarousel cards={flashcards} />
        </div>
      )}
    </main>
  );
}
