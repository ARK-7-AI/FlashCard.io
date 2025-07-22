"use client";

import Slider from "react-slick";
import "slick-carousel/slick/slick.css";
import "slick-carousel/slick/slick-theme.css";

const gradients = [
  "from-pink-200 to-pink-100",
  "from-purple-200 to-purple-100",
  "from-blue-200 to-blue-100",
  "from-yellow-200 to-yellow-100",
  "from-green-200 to-green-100",
];

/**
 * @typedef {Object} Flashcard
 * @property {string} question
 * @property {string} answer
 */

/**
 * @param {{ cards: Array<{ question: string, answer: string }> }} props
 */
export default function FlashcardCarousel({ cards }) {
  const settings = {
    dots: true,
    infinite: true,
    speed: 600,
    slidesToShow: 1,
    slidesToScroll: 1,
    swipeToSlide: true,
    arrows: true,
  };

  return (
    <Slider {...settings}>
      {cards.map((card, index) => (
        <div key={index} className="p-4">
          <div
            className={`rounded-3xl shadow-xl p-8 transition-all duration-300 bg-gradient-to-br ${
              gradients[index % gradients.length]
            } backdrop-blur-sm`}
          >
            <h3 className="text-2xl font-semibold mb-4 text-gray-800">
              Q{index + 1}: {card.question}
            </h3>
            <p className="text-lg text-gray-700">A: {card.answer}</p>
          </div>
        </div>
      ))}
    </Slider>
  );
}
