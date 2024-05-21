export const dynamic = 'force-dynamic' // defaults to auto

import { serialize } from 'cookie'
import type { NextApiRequest, NextApiResponse } from 'next'
import { encrypt } from '@/app/lib/session'
import { cookies } from 'next/headers'


export async function GET(request: NextApiRequest, response: NextApiResponse) {
  const cookieStore = cookies()
  const token = cookieStore.get('token')
  if (!token) {
    return Response.json({'message': 'No token found'}, { status: 401 })
  }
  return Response.json({'message': 'set cookie'}, {
    status: 200,
    headers: { 'Set-Cookie': `token=${token.value}` },
  })
}
