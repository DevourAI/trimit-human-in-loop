'use client';
import React, {
  createContext,
  ReactNode,
  useContext,
  useEffect,
  useState,
} from 'react';

import { useUser } from '@/contexts/user-context';
import { getUploadedVideos } from '@/lib/api';
import {
  UploadedVideo
} from '@/gen/openapi/api';

interface UserVideosData {
  videos: UploadedVideo[];
}

interface UserVideosDataContextProps {
  userVideosData: UserVideosData;
  setUserVideosData: React.Dispatch<React.SetStateAction<UserVideosData>>;
  isLoading: boolean;
}

const UserVideosDataContext = createContext<UserVideosDataContextProps | undefined>(undefined);

export const UserVideosDataProvider = ({ children }: { children: ReactNode }) => {
  const { userData } = useUser();
  const [userVideosData, setUserVideosData] = useState<UserVideosData>({
    videos: []
  });
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchUploadedVideos = async () => {
      setIsLoading(true);
      try {
        const videos = await getUploadedVideos({
          user_email: userData.email,
        } as GetUploadedVideoParams);

        setUserVideosData({videos: videos as UploadedVideo[]});
      } catch (error) {
        console.error('Error fetching uploaded videos:', error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchUploadedVideos();
  }, [userData]);

  return (
    <UserVideosDataContext.Provider
      value={{ userVideosData, setUserVideosData, isLoading }}
    >
      {children}
    </UserVideosDataContext.Provider>
  );
};

export const useUserVideosData = () => {
  const context = useContext(UserVideosDataContext);
  if (context === undefined) {
    throw new Error('useUserVideosData must be used within a UserVideosDataProvider');
  }
  return context;
};
