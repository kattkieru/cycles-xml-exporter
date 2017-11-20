
import random
import math
import os
import xml.etree.ElementTree as etree
import xml.dom.minidom as dom

import bpy
import bpy.types
import mathutils
from mathutils import Matrix, Vector

_options = {}
def export_cycles(fp, scene, inline_textures=False):
	global _options
	_options = {
		'inline_textures': inline_textures
	}

	## make sure to wrap with the proper root tag
	fp.write('<cycles>\n')

	for node in gen_scene_nodes(scene):
		if node is not None:
			write(node, fp)

	fp.write('</cycles>\n')

	return {'FINISHED'}

def gen_scene_nodes(scene):
	yield write_film(scene)
	written_materials = set()
	written_light_shaders = set()

	yield write_material(scene.world, 'background')

	for ob in scene.objects:
		if ob.type == 'LAMP':
			if ob.data.node_tree and not hash(ob.data.node_tree) in written_light_shaders:
				material_node = write_material( ob.data )
				if material_node:
					written_light_shaders.add(hash(ob.data.node_tree))
					yield material_node
		else:
			materials = getattr(ob.data, 'materials', []) or getattr(ob, 'materials', [])
			for material in materials:
				if hash(material) not in written_materials:
					material_node = write_material(material)
					if material_node:
						written_materials.add(hash(material))
						yield material_node

		yield  write_object(ob, scene=scene)


def write_camera(camera, scene):
	camera = camera.data

	if camera.type == 'ORTHO':
		camera_type = 'orthogonal'
	elif camera.type == 'PERSP':
		camera_type = 'perspective'
	else:
		raise Exception('Camera type %r unknown!' % camera.type)

	return etree.Element('camera', {
		'type': camera_type,

		# fabio: untested values. assuming to be the same as found here:
		# http://www.blender.org/documentation/blender_python_api_2_57_release/bpy.types.Camera.html#bpy.types.Camera.clip_start
		'nearclip': str(camera.clip_start),
		'farclip': str(camera.clip_end),
		'focaldistance': str(camera.dof_distance),
		'fov': str( math.radians(camera.lens) ),
	})


def write_film(scene):
	render = scene.render
	scale = scene.render.resolution_percentage / 100.0
	size_x = int(scene.render.resolution_x * scale)
	size_y = int(scene.render.resolution_y * scale)

	return etree.Element('film', {'width': str(size_x), 'height': str(size_y)})



def write_object(object, scene):
	if object.type == 'MESH':
		node = write_mesh(object, scene)
	elif object.type == 'LAMP':
		node = write_light(object)
	elif object.type == 'CAMERA':
		node = write_camera(object, scene)
	else:
		raise NotImplementedError('Object type: %r' % object.type)

	node = wrap_in_state(node, object)
	node = wrap_in_transforms(node, object)
	return node


# from the Node Wrangler, by Barte
def write_material( material, tag_name='shader'):
	light_types       = { 'POINT', 'SUN', 'SPOT', 'HEMI', 'AREA' }
	node_output_types = { 'OUTPUT', 'OUTPUT_MATERIAL', 'OUTPUT_WORLD', 'OUTPUT_LAMP' }

	did_copy = False
	is_light_shader = (material.type in light_types) if not isinstance(material, bpy.types.World) else False

	# print("|==+ Processing material {}".format(material.name) )

	if not is_light_shader and not material.use_nodes:
		# print("  |+ Copying material")
		did_copy = True
		material = material.copy()
		material.use_nodes = True

	def xlateSocket(typename, socketname):
		for i in xlate:
			if i[0]==typename:
				for j in i[2]:
					if j[0]==socketname:
						return j[1]
		return socketname
	
	def xlateType(typename ):
		for i in xlate:
			if i[0]==typename:
				return i[1]
		return typename.lower()
	
	def isConnected(socket, links):
		for link in links:
			if link.from_socket == socket or link.to_socket == socket:
				return True
		return False

	def is_output(node):
		return node.type in node_output_types

	def socketIndex(node, socket):
		socketindex=0
		countname=0
		for i in node.inputs:
			if i.name == socket.name:
				countname += 1
				if i==socket:
					socketindex=countname
		if socketindex>0:
			if countname>1:
				return str(socketindex)
			else:
				return ''
		countname=0
		for i in node.outputs:
			if i.name == socket.name:
				countname += 1
				if i==socket:
					socketindex=countname
		if socketindex>0:
			if countname>1:
				return str(socketindex)
		return ''
	#           blender        <--->     cycles
	xlate = ( ("RGB",                   "color",()),
			  ("BSDF_DIFFUSE",          "diffuse_bsdf",()),
			  ("BSDF_TRANSPARENT",      "transparent_bsdf",()),
			  ("BUMP",                  "bump",()),
			  ("FRESNEL",               "fresnel",()),
			  ("MATH",                  "math",()),
			  ("MIX_RGB",               "mix",()),
			  ("MIX_SHADER",            "mix_closure",(("Shader","closure"),)),
			  ("OUTPUT_MATERIAL",       "",()),
			  ("OUTPUT_LAMP",           "",()),
			  ("SUBSURFACE_SCATTERING", "subsurface_scattering",()),
			  ("TEX_IMAGE",             "image_texture",()),
			  ("TEX_MAGIC",             "magic_texture",()),
			  ("TEX_NOISE",             "noise_texture",()),
			  ("TEX_COORD",             "texture_coordinate",()),
			)
	
	node_tree = material.node_tree
	# nodes, links = get_nodes_links(context)
	nodes, links = node_tree.nodes, node_tree.links

	output_nodes = list(filter(is_output, nodes))

	if not output_nodes:
		# print("  |+ no output nodes")
		return None

	nodes = list(nodes)  # We don't want to remove the node from the actual scene.
	nodes.remove(output_nodes[0])

	# print( "  |+ Translating {} nodes in shader...".format(len(nodes)) )

	if is_light_shader:
		shader_name = '_'.join( ('lamp',material.name) )
	else:
		shader_name = material.name

	# print( "  |+ Shader name: {}".format(shader_name) )

	# tag_name is usually 'shader' but could be 'background' for world shaders
	shader = etree.Element(tag_name, { 'name': shader_name })
	
	def socket_name(socket, node):
		# TODO don't do this. If it has a space, don't trust there's
		# no other with the same name but with underscores instead of spaces.
		return xlateSocket(node.type, socket.name.replace(' ', '')) + socketIndex(node, socket)
	
	def shader_node_name(node):
		if is_output(node):
			return 'output'

		return node.name.replace(' ', '_')

	def special_node_attrs(node):
		def image_src(image):
			path = node.image.filepath_raw
			if path.startswith('//'):
				path = path[2:]

			if _options['inline_textures']:
				return { 'src': path }
			else:
				import base64
				w, h = image.size
				image = image.copy()
				newimage = bpy.data.images.new('/tmp/cycles_export', width=w, height=h)
				newimage.file_format = 'PNG'
				newimage.pixels = [pix for pix in image.pixels]
				newimage.filepath_raw = '/tmp/cycles_export'
				newimage.save()
				with open('/tmp/cycles_export', 'rb') as fp:
					return {
						'src': path,
						'inline': base64.b64encode(fp.read()).decode('ascii')
					}
			
		if node.type == 'TEX_IMAGE' and node.image is not None:
			return image_src(node.image)
		elif node.type == 'RGB':
			color = space_separated_float3(
				node.outputs['Color']
					.default_value[:3])

			return { 'value': color }
		elif node.type == 'VALUE':
			return {
				'value': '%f' % node.outputs['Value'].default_value
			}

		return {}

	connect_later = []

	def gen_shader_node_tree():
		# print( "  |+ Processing Nodes..." )

		for node in nodes:
			name = shader_node_name( node )
			# print( "    |+ {}...".format(name) )
			node_attrs = { 'name': name }
			node_name = xlateType(node.type)

			for input in node.inputs:
				if isConnected(input,links):
					continue
				if not hasattr(input,'default_value'):
					continue

				el = None
				sock = None
				if input.type == 'RGBA':
					el = etree.Element('color', {
						'value': '%f %f %f' % (
							input.default_value[0],
							input.default_value[1],
							input.default_value[2],
						)
					})
					sock = 'Color'
				elif input.type == 'VALUE':
					el = etree.Element('value', { 'value': '%f' % input.default_value })
					sock = 'Value'
				elif input.type == 'VECTOR':
					pass  # TODO no mapping for this?
				else:
					print('TODO: unsupported default_value for socket of type: %s', input.type);
					print('(node %s, socket %s)' % (node.name, input.name))
					continue

				if el is not None:
					el.attrib['name'] = input.name + ''.join(
						random.choice('abcdef')
						for x in range(5))

					connect_later.append((
						el.attrib['name'],
						sock,
						node,
						input
					))
					yield el

			node_attrs.update(special_node_attrs(node))
			yield etree.Element(node_name, node_attrs)


	# print( "  |+ Traversing tree..." )
	for snode in gen_shader_node_tree():
		if snode is not None:
			shader.append(snode)

	# print( "  |+ Processing links..." )
	for link in links:
		from_node = shader_node_name(link.from_node)
		to_node = shader_node_name(link.to_node)

		from_socket = socket_name(link.from_socket, node=link.from_node)
		to_socket = socket_name(link.to_socket, node=link.to_node)

		shader.append(etree.Element('connect', {
			'from': '%s %s' % (from_node, from_socket.replace(' ', '_')),
			'to': '%s %s' % (to_node, to_socket.replace(' ', '_')),

			# uncomment to be compatible with the new proposed syntax for defining nodes
			# 'from_node': from_node,
			# 'to_node': to_node,
			# 'from_socket': from_socket,
			# 'to_socket': to_socket
		}))

	# print( "  |+ Processing connect_later..." )

	for fn, fs, tn, ts in connect_later:
		from_node = fn
		to_node = shader_node_name(tn)

		from_socket = fs
		to_socket = socket_name(ts, node=tn)

		shader.append(etree.Element('connect', {
			'from': '%s %s' % (from_node, from_socket.replace(' ', '_')),
			'to': '%s %s' % (to_node, to_socket.replace(' ', '_')),

			# uncomment to be compatible with the new proposed syntax for defining nodes
			# 'from_node': from_node,
			# 'to_node': to_node,
			# 'from_socket': from_socket,
			# 'to_socket': to_socket
		}))

	if did_copy:
		# print("    |+ Removing did_copy duplicate.")
		# TODO delete the material we created as a hack to support materials with use_nodes == False
		pass

	print( "  |+ Successfully translated shader for: {}".format(material.name) )
	return shader


def write_light(ob):
	def FF( f ):
		return '{}'.format( f )

	def II( i ):
		return '{}'.format( int(i) )

	def BB( b ):
		return '{}'.format( 1 if b else 0 )

	## these customizations are taken from intern/cycles/blender/blender_object.cpp,
	## BlenderSync::sync_light().

	type_conversion = {
		'SUN':   'distant',
		'POINT': 'point',
		'HEMI':  'distant',
		'AREA':  'area',
		'SPOT':  'spot',
	}

	cscene = bpy.context.scene.cycles
	clamp  = ob.data.cycles
	clvis  = ob.cycles_visibility

	samples = clamp.samples
	if cscene.use_square_samples:
		samples = samples * samples

	node = etree.Element(
		'light', {
			'co'              : '{} {} {}'.format( *ob.location ),
			'type'            : type_conversion[ob.data.type],
			'dir'             : '{} {} {}'.format( *(Vector([0,0,-1]) * ob.matrix_world) ),
			'cast_shadow'     : BB( clamp.cast_shadow ),
			'use_mis'         : BB( clamp.use_multiple_importance_sampling ),
			'samples'         : FF( samples ),
			'max_bounces'     : II( clamp.max_bounces ),
			'use_diffuse'     : BB( clvis.diffuse ),
			'use_glossy'      : BB( clvis.glossy ),
			'use_transmission': BB( clvis.transmission ),
			'use_scatter'     : BB( clvis.scatter ),
		}
	)

	if ob.data.type in { 'POINT', 'SPOT', 'SUN' }:
		node.set( 'size', FF(ob.data.shadow_soft_size) )

	if ob.data.type == 'SPOT':
		node.set( 'spot_angle', FF(ob.data.spot_size) )
		node.set( 'spot_smooth', FF(ob.data.spot_blend) )

	elif ob.data.type == 'HEMI':
		## these seem to be an odd type of distant light
		node.set( 'size', "0.0" )
	
	elif ob.data.type == 'AREA':
		node.set( 'size', "1.0" )
		node.set( 'axisu', '{} {} {}'.format(*(Vector([1,0,0]) * ob.matrix_world)) )
		node.set( 'axisv', '{} {} {}'.format(*(Vector([0,1,0]) * ob.matrix_world)) )
		node.set( 'sizeu', FF(ob.data.size) )
		node.set( 'sizev', FF(ob.data.size_y if ob.data.shape == 'RECTANGLE' else ob.data.size) )
		if clamp.is_portal:
			node.set( 'is_portal', '1' )

	return node


def write_mesh(object, scene):
	mesh = object.to_mesh(scene, True, 'PREVIEW')

	# generate mesh node
	nverts = ""
	verts = ""

	P = ' '.join(space_separated_float3(v.co) for v in mesh.vertices)

	for i, f in enumerate(mesh.tessfaces):
		nverts += str(len(f.vertices)) + " "

		for v in f.vertices:
			verts += str(v) + " "

		verts += " "

	return etree.Element('mesh', attrib={'nverts': nverts, 'verts': verts, 'P': P})

def wrap_in_transforms(xml_element, object):
	matrix = object.matrix_world

	if (object.type == 'CAMERA'):
		# In cycles, the camera points at its Z axis
		## kiki edit: not entirely sure why the camera ends up flipped, but
		## rotating around Z by 180 solves it
		rot = mathutils.Matrix.Rotation(math.pi, 4, 'X')
		correction = mathutils.Matrix.Rotation(math.pi, 4, 'Z')
		matrix = matrix.copy() * rot * correction

	wrapper = etree.Element('transform', { 'matrix': space_separated_matrix(matrix.transposed()) })
	wrapper.append(xml_element)

	return wrapper

def wrap_in_state(xml_element, ob):
	# UNSUPPORTED: Meshes with multiple materials
	## light shader support
	shader_name = None

	if ob.type == 'LAMP' and ob.data.node_tree:
		shader_name = '_'.join( ('lamp', ob.data.name) )
		print("Assigning lamp shader {}.".format(shader_name))
	else:
		try:
			material = getattr(ob.data, 'materials', [])[0]
			shader_name = material.name
		except LookupError:
			pass

	if not shader_name:
		return xml_element

	state = etree.Element('state', {
		'shader': shader_name
	})

	state.append( xml_element )
	return state

def space_separated_float3(coords):
	float3 = list(map(str, coords))
	assert len(float3) == 3, 'tried to serialize %r into a float3' % float3
	return ' '.join(float3)

def space_separated_float4(coords):
	float4 = list(map(str, coords))
	assert len(float4) == 4, 'tried to serialize %r into a float4' % float4
	return ' '.join(float4)

def space_separated_matrix(matrix):
	return ' '.join(space_separated_float4(row) + ' ' for row in matrix)

def write(node, fp):
	# strip(node)
	s = etree.tostring(node, encoding='unicode')
	# s = dom.parseString(s).toprettyxml()
	fp.write(s)
	fp.write('\n')

