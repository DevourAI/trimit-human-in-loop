"use client";
import { cn } from "@/lib/utils"
import useSWR from 'swr'
import { Button, buttonVariants } from "@/components/ui/button"
import { googleLogout, GoogleLogin } from '@react-oauth/google';
import React, { createContext, useContext, useState } from 'react';
import { GoogleOAuthProvider } from '@react-oauth/google';

const clientId = process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID;

const fetcher = (url) => fetch(url).then((res) => res.json());
const fetchWithToken = (url, token) => {
  const headers = {
    Authorization: `Bearer ${token}`,
  };
  return fetch(url, {headers: headers}).then((res) => res.json());
}
const fetchWithBody = (url, param) => {
  return fetch(url, {body: param}).then((res) => res.json());
}
const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;

const getProtectedEndpoint = (url: string, token: string) => {
  const { data, error, isLoading } = useSWR(
    [`${baseUrl}${url}`, token],
    ([url, token]) => fetchWithToken(url, token),
  )
  if (!error && !isLoading) {
    console.log(data);
    return data;
  } else {
    console.log(error, isLoading);
  }
}

export default function Login() {
  const { data, error, isLoading } = useSWR("/cookies", fetcher);
  let initialToken = "";
  if (data && data.token && data.token.value) {
    initialToken = data.token.value;
  }
  const [token, setToken] = useState(initialToken)
  const logout = () => {
    googleLogout();
    setToken("");
  }
  let loggedIn = token ? true : false;

  return (
    <div>
        <GoogleOAuthProvider clientId={clientId}>
            <Button onClick={() => getProtectedEndpoint("/getUserData")} className={cn(buttonVariants())}>
              Get User Data
            </Button>

            { loggedIn?
              <Button onClick={() => logout()} className={cn(buttonVariants())}>
                Logout
              </Button>
            :
                <GoogleLogin
                   onSuccess={credentialResponse => {
                     setToken(credentialResponse.credential);
                     fetch(`/login?token=${credentialResponse.credential}`).then((res) => {});
                   }}
                   onError={() => {
                     console.log('Login Failed');
                   }}
                   useOneTap
                 />
            }
        </GoogleOAuthProvider>
     </div>
   )
}
