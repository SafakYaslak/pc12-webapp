import React, { createContext, useContext, useState } from 'react';

interface TabsContextType {
  value: string;
  onValueChange: (value: string) => void;
}

const TabsContext = createContext<TabsContextType | undefined>(undefined);

interface TabsProps {
  value: string;
  onValueChange: (value: string) => void;
  children: React.ReactNode;
}

export const Tabs: React.FC<TabsProps> = ({ value, onValueChange, children }) => {
  return (
    <TabsContext.Provider value={{ value, onValueChange }}>
      <div className="tabs-root">{children}</div>
    </TabsContext.Provider>
  );
};

interface TabsListProps {
  children: React.ReactNode;
  className?: string;
}

export const TabsList: React.FC<TabsListProps> = ({ children, className = '' }) => {
  return (
    <div className={`flex space-x-2 border-b border-gray-200 ${className}`}>
      {children}
    </div>
  );
};

interface TabsTriggerProps {
  value: string;
  children: React.ReactNode;
  className?: string;
  disabled?: boolean;
}

export const TabsTrigger: React.FC<TabsTriggerProps> = ({
  value,
  children,
  className = '',
  disabled = false,
}) => {
  const context = useContext(TabsContext);
  
  if (!context) {
    throw new Error('TabsTrigger must be used within a Tabs component');
  }
  
  const { value: selectedValue, onValueChange } = context;
  const isSelected = selectedValue === value;
  
  return (
    <button
      type="button"
      role="tab"
      aria-selected={isSelected}
      disabled={disabled}
      onClick={() => onValueChange(value)}
      className={`
        px-4 py-2 text-sm font-medium transition-all duration-200
        ${isSelected 
          ? 'border-b-2 border-blue-500 text-blue-600' 
          : 'text-gray-500 hover:text-gray-700 hover:border-b-2 hover:border-gray-300'
        }
        ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
        ${className}
      `}
    >
      {children}
    </button>
  );
};

interface TabsContentProps {
  value: string;
  children: React.ReactNode;
  className?: string;
}

export const TabsContent: React.FC<TabsContentProps> = ({
  value,
  children,
  className = '',
}) => {
  const context = useContext(TabsContext);
  
  if (!context) {
    throw new Error('TabsContent must be used within a Tabs component');
  }
  
  const { value: selectedValue } = context;
  const isSelected = selectedValue === value;
  
  if (!isSelected) {
    return null;
  }
  
  return (
    <div
      role="tabpanel"
      className={`py-4 ${className}`}
    >
      {children}
    </div>
  );
};