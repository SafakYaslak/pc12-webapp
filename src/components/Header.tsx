import React, { useState, useEffect } from 'react';
import logo from '../assets/ikclogo_yuarlak_beyaz.png';

const Header: React.FC = () => {
  const [isScrolled, setIsScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 10);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <header 
      className={`sticky top-0 z-50 transition-all duration-300 ${
        isScrolled 
          ? 'bg-white shadow-md py-2' 
          : 'bg-transparent py-4'
      }`}
    >
      <div className="container mx-auto px-4">
        <div className="flex justify-center items-center">
          <div className="flex items-center">
            <img 
              src={logo}
              alt="Logo"
              className="w-12 h-12 mr-3 object-contain" // Boyutlar büyütüldü
            />
            
            <h1 className="text-xl md:text-2xl font-bold text-gray-900">
              ImageAI Analysis
            </h1>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;