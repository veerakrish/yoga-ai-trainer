import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';

const SelectionOption = ({ text, isSelected, isActive, onClick }) => {
  return (
    <motion.div
      className={`p-4 m-2 rounded-lg transition-all duration-300 cursor-pointer
        ${isActive ? 'bg-blue-600' : 'bg-gray-800'}
        ${isSelected ? 'ring-4 ring-green-400 scale-110' : ''}`}
      whileHover={{ scale: 1.05 }}
      onClick={onClick}
    >
      <div className="flex items-center space-x-3">
        <div className={`w-4 h-4 rounded-full ${isSelected ? 'bg-green-400' : 'bg-gray-400'}`} />
        <span className={`text-xl font-medium ${isSelected ? 'text-white' : 'text-gray-200'}`}>
          {text}
        </span>
      </div>
    </motion.div>
  );
};

const SelectionWidget = () => {
  const [selectedLanguage, setSelectedLanguage] = useState(null);
  const [selectedExercise, setSelectedExercise] = useState(null);
  const [activeWidget, setActiveWidget] = useState('language'); // 'language' or 'exercise'
  
  const languages = [
    'English',
    'Telugu',
    'Tamil',
    'Hindi',
    'Kannada',
    'Malayalam'
  ];
  
  const exercises = [
    'Surya Namaskar',
    'Vinyasana Yoga',
    'Hatha Yoga'
  ];

  useEffect(() => {
    if (selectedLanguage) {
      setActiveWidget('exercise');
    }
  }, [selectedLanguage]);

  return (
    <div className="w-full max-w-4xl mx-auto p-8 bg-gray-900 rounded-xl">
      {/* Header */}
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-white mb-4">TrainWithAI</h1>
        <p className="text-gray-400">Use hand gestures to make your selection</p>
      </div>

      {/* Language Selection */}
      <div className={`mb-8 transition-opacity duration-300 ${activeWidget === 'language' ? 'opacity-100' : 'opacity-50'}`}>
        <h2 className="text-2xl font-bold text-white mb-4">Select Language</h2>
        <div className="grid grid-cols-2 gap-4">
          {languages.map((lang) => (
            <SelectionOption
              key={lang}
              text={lang}
              isSelected={lang === selectedLanguage}
              isActive={activeWidget === 'language'}
              onClick={() => setSelectedLanguage(lang)}
            />
          ))}
        </div>
      </div>

      {/* Exercise Selection */}
      <div className={`transition-opacity duration-300 ${activeWidget === 'exercise' ? 'opacity-100' : 'opacity-50'}`}>
        <h2 className="text-2xl font-bold text-white mb-4">Select Exercise</h2>
        <div className="grid grid-cols-1 gap-4">
          {exercises.map((exercise) => (
            <SelectionOption
              key={exercise}
              text={exercise}
              isSelected={exercise === selectedExercise}
              isActive={activeWidget === 'exercise'}
              onClick={() => setSelectedExercise(exercise)}
            />
          ))}
        </div>
      </div>

      {/* Gesture Instructions */}
      <div className="mt-8 p-4 bg-gray-800 rounded-lg">
        <h3 className="text-xl font-bold text-white mb-2">Gesture Controls</h3>
        <ul className="text-gray-300 space-y-2">
          <li>👆 Point left/right: Navigate options</li>
          <li>✋ Open palm: Select current option</li>
          <li>👊 Closed fist: Go back</li>
        </ul>
      </div>

      {/* Current Selection Display */}
      <div className="mt-6 p-4 bg-blue-900 rounded-lg">
        <h3 className="text-lg font-medium text-white mb-2">Current Selection</h3>
        <p className="text-gray-200">Language: {selectedLanguage || 'Not selected'}</p>
        <p className="text-gray-200">Exercise: {selectedExercise || 'Not selected'}</p>
      </div>
    </div>
  );
};

export default SelectionWidget;