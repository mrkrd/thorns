#include <stdio.h>

#include "Python.h"
#include "numpy/arrayobject.h"



static PyObject*
c_signal_to_spikes(PyObject* self, PyObject* args)
{
/* Input singal */
     PyObject *signal_arg;
     PyObject *signal_arr;
     double *signal_data;
     int signal_len;
     npy_intp *spikes_dim;

     /* Output spikes */
     PyObject *spikes_arr;
     int spikes_nd = 1;
     npy_intp spikes_dims[1];
     double *spikes_data;
     int spikes_len;
     int spikes_idx;

     /* Temp array */
     PyObject *sum;
     int i,j;

     if (!PyArg_ParseTuple(args, "O", &signal_arg))
	  return NULL;


     /* Unpack input array */
     signal_arr = PyArray_FROM_OTF(signal_arg, NPY_DOUBLE, NPY_IN_ARRAY);
     if (signal_arr == NULL) return NULL;
     signal_data = PyArray_DATA(signal_arr);


     /* Generate output array */
     signal_len = PyArray_DIMS(signal_arr)[0];
     spikes_len = 0;
     for (i = 0; i < signal_len; i++) {
	  spikes_len += signal_data[i];
     }
     spikes_dims[0] = spikes_len;
     spikes_arr = PyArray_SimpleNew(spikes_nd, spikes_dims, NPY_DOUBLE);
     spikes_data = PyArray_DATA(spikes_arr);


     /* Convert signal to events */
     spikes_idx = 0;
     for (i = 0; i < signal_len; i++) {
	  for (j = 0; j < signal_data[i]; j++) {
	       spikes_data[spikes_idx] = i;
	       spikes_idx++;
	  }
     }


     Py_DECREF(signal_arr);
     return spikes_arr;
}


static PyMethodDef
cThorns_Methods[] =
{
     {"c_signal_to_spikes", c_signal_to_spikes, METH_VARARGS, "Convert signal to events."},
     {NULL, NULL, 0, NULL}
};



PyMODINIT_FUNC
init_cThorns(void)
{
     (void)Py_InitModule("_cThorns", cThorns_Methods);
     import_array();
}
