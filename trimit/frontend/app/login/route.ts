export const dynamic = 'force-dynamic' // defaults to auto

import { serialize } from 'cookie'
import type { NextApiRequest, NextApiResponse } from 'next'
import { encrypt } from '@/app/lib/session'

import { cookies } from 'next/headers'

export async function GET(request: Request) {
  const token = request.nextUrl.searchParams.get('token')
  const cookieStore = cookies()
  cookieStore.set('token', token)
  return Response.json({'message': 'stored token in cookieStore'}, {
    status: 200,
    headers: { 'Set-Cookie': `token=${token}` },
  })
}
