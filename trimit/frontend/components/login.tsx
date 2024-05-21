"use client";
import { jwtDecode } from "jwt-decode";
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
    return data;
  } else {
    console.log(error, isLoading);
  }
}

export default function Login() {
  const { data, error, isLoading } = useSWR("/userData", fetcher);
  let emptyUserData = {'email': '', 'name': '', 'picture': ''};
  let initialUserData = emptyUserData;
  if (data && data.userData && data.userData.value) {
    initialUserData = data.userData.value;
  }
  const [userData, setUserData] = useState(initialUserData)
  const logout = () => {
    googleLogout();
    setUserData(emptyUserData);
  }
  let loggedIn = userData.email ? true : false;

  return (
    <div>
        <GoogleOAuthProvider clientId={clientId}>

            { loggedIn?
            <h1>Welcome {userData.given_name}</h1>
            : null
            }

            { loggedIn?
              <Button onClick={() => logout()} className={cn(buttonVariants())}>
                Logout
              </Button>
            :
                <GoogleLogin
                   onSuccess={credentialResponse => {
                     const decoded = jwtDecode(credentialResponse.credential);
                     setUserData(decoded);
                     fetch(`/userData`, {
                       body: JSON.stringify({userData: decoded}),
                       method: 'POST',
                       headers: {
                         'Content-Type': 'application/json'
                       }
                     }).then((res) => {});
                   }}
                   onError={() => {
                     console.log('Login Failed');
                   }}
                   useOneTap
                   auto_select
                 />
            }
        </GoogleOAuthProvider>
     </div>
   )
}
