import React from 'react';

const Footer: React.FC = () => {
  return (
    <footer className="bg-gray-900 text-gray-300 py-8">
      <div className="container mx-auto px-4">
        <div className="flex justify-center items-center h-32"> {/* Yükseklik ve ortala */}
          <h1 className="text-4xl font-bold text-white uppercase tracking-wider">
            ŞAFAK YAŞLAK
          </h1>
        </div>
      </div>
    </footer>
  );
};

export default Footer;