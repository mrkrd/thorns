#include "Python.h"
#include "numpy/arrayobject.h"



static PyObject*
_c_signal_to_spikes(PyObject* self, PyObject* args)
{
     /* Input singal */
     PyObject *signal_arg;
     PyObject *signal_arr;
     double *signal_data;

     /* Output spikes */
     PyObject *spikes;
     int nd = 1;
     npy_intp dims[1];
     double *spikes_data;

     /* Temp array */
     PyObject *sum;


     if (!PyArg_ParseTuple(args, "O", &signal_arg))
	  return NULL;

     /* Unpack input array */
     signal_arr = PyArray_FROM_OTF(signal_arg, NPY_DOUBLE, NPY_IN_ARRAY);
     if (signal_arr == NULL) return NULL;

     /* Generate output array */
     spikes_len = 0;
     for (i = 0; i < signal_len; i++) {
	  spikes_len += signal_data[i];
     }
     dims[0] = spikes_len;
     spikes_arr = PyArray_SimpleNew(nd, dims, NPY_DOUBLE);
     spikes_data =



     Py_RETURN_NONE;
}


static PyMethodDef
cThorns_Methods[] =
{
     {"_c_signal_to_spikes", _c_signal_to_spikes, METH_VARARGS, "Convert signal to events."},
     {NULL, NULL, 0, NULL}
};



PyMODINIT_FUNC
init_cThorns(void)
{
     (void)Py_InitModule("_cThorns", DSAM_Methods);
     import_array();
}

