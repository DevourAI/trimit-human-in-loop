// contexts/user-context.tsx
'use client';
import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  ReactNode,
} from 'react';
import Cookies from 'js-cookie';
import { jwtDecode } from 'jwt-decode';

interface UserData {
  email: string;
  name: string;
  picture: string;
  given_name?: string;
}

interface UserContextProps {
  userData: UserData;
  setUserData: React.Dispatch<React.SetStateAction<UserData>>;
  isLoggedIn: boolean;
  isLoading: boolean;
  login: (token: string) => void;
  logout: () => void;
}

const UserContext = createContext<UserContextProps | undefined>(undefined);

export const UserProvider = ({ children }: { children: ReactNode }) => {
  const [userData, setUserData] = useState<UserData>({
    email: '',
    name: '',
    picture: '',
  });
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const token = Cookies.get('token');
    if (token) {
      try {
        const decoded: UserData = jwtDecode(token);
        setUserData(decoded);
      } catch (error) {
        // If token is invalid or expired, remove it
        Cookies.remove('token');
      }
    }
    setIsLoading(false);
  }, []);

  const login = (token: string) => {
    try {
      const decoded: UserData = jwtDecode(token);
      setUserData(decoded);
      Cookies.set('token', token, { expires: 7 });
    } catch (error) {
      console.error('Invalid token', error);
    }
  };

  const logout = () => {
    setUserData({ email: '', name: '', picture: '' });
    Cookies.remove('token');
  };

  const isLoggedIn = !!userData.email;

  return (
    <UserContext.Provider
      value={{ userData, setUserData, isLoggedIn, isLoading, login, logout }}
    >
      {children}
    </UserContext.Provider>
  );
};

export const useUser = () => {
  const context = useContext(UserContext);
  if (context === undefined) {
    throw new Error('useUser must be used within a UserProvider');
  }
  return context;
};
