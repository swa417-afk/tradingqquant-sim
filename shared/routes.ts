import { z } from 'zod';
import { insertRunSchema, runs, datasets } from './schema';

// ============================================
// SHARED ERROR SCHEMAS
// ============================================
export const errorSchemas = {
  validation: z.object({
    message: z.string(),
    field: z.string().optional(),
  }),
  notFound: z.object({
    message: z.string(),
  }),
  internal: z.object({
    message: z.string(),
  }),
};

// ============================================
// API CONTRACT
// ============================================
export const api = {
  runs: {
    list: {
      method: 'GET' as const,
      path: '/api/runs',
      responses: {
        200: z.array(z.custom<typeof runs.$inferSelect>()),
      },
    },
    get: {
      method: 'GET' as const,
      path: '/api/runs/:id',
      responses: {
        200: z.custom<typeof runs.$inferSelect>(),
        404: errorSchemas.notFound,
      },
    },
    create: {
      method: 'POST' as const,
      path: '/api/runs',
      input: insertRunSchema,
      responses: {
        201: z.custom<typeof runs.$inferSelect>(),
        400: errorSchemas.validation,
      },
    },
    logs: {
        method: 'GET' as const,
        path: '/api/runs/:id/logs',
        responses: {
            200: z.object({ logs: z.string() }),
            404: errorSchemas.notFound,
        }
    }
  },
  datasets: {
    list: {
      method: 'GET' as const,
      path: '/api/datasets',
      responses: {
        200: z.array(z.custom<typeof datasets.$inferSelect>()),
      },
    },
    upload: {
      method: 'POST' as const,
      path: '/api/datasets',
      // input is multipart/form-data, validation handled in route
      responses: {
        201: z.custom<typeof datasets.$inferSelect>(),
        400: errorSchemas.validation,
      },
    },
    delete: {
        method: 'DELETE' as const,
        path: '/api/datasets/:id',
        responses: {
            204: z.void(),
            404: errorSchemas.notFound
        }
    }
  },
  configs: {
    list: {
        method: 'GET' as const,
        path: '/api/configs',
        responses: {
            200: z.array(z.string()),
        }
    },
    get: {
        method: 'GET' as const,
        path: '/api/configs/:name',
        responses: {
            200: z.object({ content: z.string() }),
            404: errorSchemas.notFound
        }
    },
    save: {
        method: 'POST' as const,
        path: '/api/configs/:name',
        input: z.object({ content: z.string() }),
        responses: {
            200: z.object({ success: z.boolean() }),
            500: errorSchemas.internal
        }
    }
  }
};

// ============================================
// HELPER
// ============================================
export function buildUrl(path: string, params?: Record<string, string | number>): string {
  let url = path;
  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      if (url.includes(`:${key}`)) {
        url = url.replace(`:${key}`, String(value));
      }
    });
  }
  return url;
}
